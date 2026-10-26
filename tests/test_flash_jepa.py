"""Tests of the spconv-free flash-attention encoder, JEPA and fine-tuning glue.

Runs in the neutrino-jepa environment (flash attention, no spconv) and in the
spconv environment (where the patch embedding is checked against spconv).
"""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from model.flash_jepa import FlashJEPA, MultiBlockMasker, flash_jepa_tiny
from model.flash_vit import (
    AHCAL, ECAL, FCAL, MUON, EventBatch, FlashViT, SeqInfo, SparsePatchEmbed,
    TokenSet, collate_tokens, padded_attention, varlen_attention, varlen_attn,
)
from utils import load_mae_encoder, move_obj

try:
    import spconv.pytorch as spconv
except ModuleNotFoundError:
    spconv = None

REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA = REPO_ROOT / "events_v7.0_500_npz" / "metadata_stats.pkl"
SMALL = dict(embed_dim=96, depth=2, num_heads=4)
HAS_FLASH = varlen_attn is not None and torch.cuda.is_available()


def synthetic_event(rng, n_fcal=None, n_ahcal=None, n_tracks=None):
    n_fcal = int(rng.integers(1, 400)) if n_fcal is None else n_fcal
    n_ahcal = int(rng.integers(0, 60)) if n_ahcal is None else n_ahcal
    n_tracks = int(rng.integers(0, 3)) if n_tracks is None else n_tracks
    # Clustered hits so that events occupy a few patches, as real events do.
    centre = rng.integers([0, 0, 0], [48, 48, 200])
    coords = np.clip(centre + rng.normal(0, [6, 6, 30], (n_fcal, 3)), 0, [47, 47, 199]).round()
    ahcal = rng.integers(0, [18, 18, 40], (n_ahcal, 3))
    flavour = rng.integers(0, 6)
    return {
        "coords": torch.tensor(coords, dtype=torch.float32),
        "feats": torch.tensor(rng.random((n_fcal, 1)), dtype=torch.float32),
        "ahcal_hits_coords": torch.tensor(ahcal, dtype=torch.float32),
        "ahcal_hits_feats": torch.tensor(rng.random((n_ahcal, 1)), dtype=torch.float32),
        "ecal_hits": torch.tensor(rng.random((5, 5)), dtype=torch.float32),
        "nb_muspec_tracks": torch.tensor([float(n_tracks)]),
        "muspec_info": torch.tensor(rng.normal(size=(n_tracks, 5)), dtype=torch.float32),
        "flavour_label": torch.tensor([flavour]),
        "charm_label": torch.tensor([rng.integers(0, 4)]),
        "vis_sp_momentum": torch.tensor(rng.normal(size=3), dtype=torch.float32),
        "out_lepton_momentum": torch.tensor(rng.normal(size=3), dtype=torch.float32),
        "jet_momentum": torch.tensor(rng.normal(size=3), dtype=torch.float32),
        "is_cc": torch.tensor([float(flavour < 5)]),
        "primary_vertex": torch.tensor(rng.normal(size=3), dtype=torch.float32),
    }


def synthetic_batch(seed=0, batch_size=6, **kwargs):
    rng = np.random.default_rng(seed)
    return collate_tokens([synthetic_event(rng, **kwargs) for _ in range(batch_size)], test=False)


def reference_attention(q, k, v, seq):
    out = torch.empty_like(q)
    cu = seq.cu_seqlens.tolist()
    for start, end in zip(cu[:-1], cu[1:]):
        if end > start:
            out[start:end] = F.scaled_dot_product_attention(
                q[start:end].transpose(0, 1), k[start:end].transpose(0, 1), v[start:end].transpose(0, 1),
            ).transpose(0, 1)
    return out


class FlashJEPATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    # ------------------------------------------------------------ tokenizer
    @unittest.skipIf(spconv is None, "spconv is not installed")
    def test_patch_embedding_matches_spconv_sparse_convolution(self):
        from model.common import choose_k1_k2

        device = "cuda" if torch.cuda.is_available() else "cpu"
        for grid, patch in (((48, 48, 200), (12, 12, 10)), ((18, 18, 40), (6, 6, 5))):
            embed = SparsePatchEmbed(grid, patch, 1, 64, torch.nn.LayerNorm).to(device)
            if embed.two_step:
                k1, k2 = choose_k1_k2(patch)
                reference = spconv.SparseSequential(
                    spconv.SparseConv3d(1, 16, kernel_size=k1, stride=k1, padding=0, bias=False),
                    torch.nn.LayerNorm(16), torch.nn.GELU(),
                    spconv.SparseConv3d(16, 64, kernel_size=k2, stride=k2, padding=0, bias=True),
                ).to(device)
                with torch.no_grad():
                    embed.proj1.weight.copy_(reference[0].weight.reshape(16, -1))
                    embed.norm1.weight.copy_(reference[1].weight.normal_())
                    embed.norm1.bias.copy_(reference[1].bias.normal_())
                    embed.proj2.weight.copy_(reference[3].weight.reshape(64, -1))
                    embed.proj2.bias.copy_(reference[3].bias.normal_())
            else:
                reference = spconv.SparseConv3d(1, 64, kernel_size=patch, stride=patch, padding=0, bias=True)
                reference = reference.to(device)
                with torch.no_grad():
                    embed.proj.weight.copy_(reference.weight.reshape(64, -1))
                    embed.proj.bias.copy_(reference.bias.normal_())

            rng = np.random.default_rng(1)
            # Distinct voxels: spconv does not sum coincident voxels consistently
            # (real events have none); see test_coincident_voxels_are_summed.
            coords = [np.unique(rng.integers(0, grid, (int(n), 3)), axis=0) for n in (300, 1, 50)]
            coords[1] = np.array([[grid[0], 0, 0]])                # outside the grid: dropped
            coords[2][0] = [grid[0], 1, 1]
            event = np.concatenate([np.full(len(c), b) for b, c in enumerate(coords)])
            coords = np.concatenate(coords)
            feats = torch.tensor(rng.random((len(coords), 1)), dtype=torch.float32, device=device)
            coords_t = torch.tensor(coords, device=device)
            event_t = torch.tensor(event, device=device)

            # Exact fp32 matrix products: training modules imported by other tests
            # enable TF32 (torch.set_float32_matmul_precision("high")).
            precision = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("highest")
            self.addCleanup(torch.set_float32_matmul_precision, precision)
            sparse = spconv.SparseConvTensor(
                feats, torch.cat([event_t[:, None], coords_t], 1).int(), list(grid), 3)
            expected = reference(sparse)
            tokens, token_event, token_patch = embed(coords_t, feats, event_t)

            grid_patches = embed.patch_grid
            index = expected.indices.long()
            expected_key = (index[:, 0] * embed.num_patches
                            + (index[:, 1] * grid_patches[1] + index[:, 2]) * grid_patches[2] + index[:, 3])
            order = torch.argsort(expected_key)
            torch.testing.assert_close(token_event * embed.num_patches + token_patch, expected_key[order])
            with torch.no_grad():
                torch.testing.assert_close(tokens, expected.features[order], rtol=1e-4, atol=1e-5)

    def test_coincident_voxels_are_summed(self):
        for grid, patch in (((48, 48, 200), (12, 12, 10)), ((18, 18, 40), (6, 6, 5))):
            embed = SparsePatchEmbed(grid, patch, 1, 32, torch.nn.LayerNorm)
            coords = torch.tensor([[1, 2, 3], [1, 2, 3], [7, 8, 9]])
            event = torch.zeros(3, dtype=torch.long)
            twice = embed(coords, torch.tensor([[0.3], [0.5], [0.2]]), event)[0]
            once = embed(coords[1:], torch.tensor([[0.8], [0.2]]), event[1:])[0]
            torch.testing.assert_close(twice, once)

    @unittest.skipUnless(spconv is not None and torch.cuda.is_available() and METADATA.is_file(),
                         "needs spconv, a GPU and the local events")
    def test_patch_embedding_matches_spconv_on_real_events(self):
        from dataset import SparseFASERCALMapDataset
        from model.common import choose_k1_k2
        from utils.funcs import collate

        precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        self.addCleanup(torch.set_float32_matmul_precision, precision)
        dataset = SparseFASERCALMapDataset(SimpleNamespace(
            train=True, stage1=False, preprocessing_input="log", preprocessing_output="log",
            label_smoothing=0.02, mixup_alpha=0.0, dataset_path=str(METADATA.parent),
            metadata_path=str(METADATA)))
        dataset.augmentations_enabled = True
        np.random.seed(0)
        events = [dataset[i] for i in range(0, 640, 20)]
        reference_batch, batch = collate(events, test=False), collate_tokens(events, test=False)
        for prefix, sparse_key, grid, patch in (("fcal", "x_sp", (48, 48, 200), (12, 12, 10)),
                                                ("ahcal", "ahcal_x_sp", (18, 18, 40), (6, 6, 5))):
            embed = SparsePatchEmbed(grid, patch, 1, 128, torch.nn.LayerNorm).cuda()
            if embed.two_step:
                k1, k2 = choose_k1_k2(patch)
                reference = spconv.SparseSequential(
                    spconv.SparseConv3d(1, 32, k1, k1, bias=False), torch.nn.LayerNorm(32), torch.nn.GELU(),
                    spconv.SparseConv3d(32, 128, k2, k2, bias=True)).cuda()
                with torch.no_grad():
                    embed.proj1.weight.copy_(reference[0].weight.reshape(32, -1))
                    embed.proj2.weight.copy_(reference[3].weight.reshape(128, -1))
                    embed.proj2.bias.copy_(reference[3].bias.normal_())
            else:
                reference = spconv.SparseConv3d(1, 128, patch, patch, bias=True).cuda()
                with torch.no_grad():
                    embed.proj.weight.copy_(reference.weight.reshape(128, -1))
                    embed.proj.bias.copy_(reference.bias.normal_())
            sparse = reference_batch[sparse_key]
            sparse = spconv.SparseConvTensor(sparse.features.cuda(), sparse.indices.cuda(),
                                             sparse.spatial_shape, sparse.batch_size)
            with torch.no_grad():
                expected = reference(sparse)
                tokens, event, patch_index = embed(*(batch[f"{prefix}_{key}"].cuda()
                                                     for key in ("coords", "feats", "event")))
            g = embed.patch_grid
            index = expected.indices.long()
            key = index[:, 0] * embed.num_patches + (index[:, 1] * g[1] + index[:, 2]) * g[2] + index[:, 3]
            order = torch.argsort(key)
            # The spconv collate adds one dummy origin voxel to events without AHCAL hits.
            common = torch.isin(key[order], event * embed.num_patches + patch_index)
            torch.testing.assert_close(key[order][common], event * embed.num_patches + patch_index)
            torch.testing.assert_close(tokens, expected.features[order][common], rtol=1e-4, atol=1e-5)

    def test_tokens_are_grouped_by_event_and_kind(self):
        batch = synthetic_batch(batch_size=5)
        encoder = FlashViT(metadata=None, **SMALL)
        tokens = encoder.embed(batch)
        self.assertTrue(bool((tokens.event[1:] >= tokens.event[:-1]).all()))
        for b in range(5):
            kinds = tokens.kind[tokens.event == b]
            self.assertTrue(bool((kinds[1:] >= kinds[:-1]).all()))
            self.assertEqual(int((kinds == ECAL).sum()), 1)
            self.assertEqual(int((kinds == MUON).sum()), 1)
            coords = batch["fcal_coords"][batch["fcal_event"] == b].long()
            occupied = torch.unique(coords // torch.tensor([12, 12, 10]), dim=0)
            self.assertEqual(int((kinds == FCAL).sum()), len(occupied))

    def test_events_do_not_interact(self):
        rng = np.random.default_rng(3)
        events = [synthetic_event(rng) for _ in range(4)]
        encoder = FlashViT(metadata=None, **SMALL).eval()
        with torch.no_grad():
            together = encoder.forward_features(collate_tokens(events, test=False))
            alone = encoder.forward_features(collate_tokens(events[2:3], test=False))
        tokens, seq = together
        torch.testing.assert_close(tokens[seq.event == 2], alone[0], rtol=1e-5, atol=1e-5)

    # ------------------------------------------------------------ attention
    def test_padded_attention_is_exact_per_event_attention(self):
        event = torch.repeat_interleave(torch.arange(4), torch.tensor([3, 0, 7, 1]))
        seq = SeqInfo.from_event(event, 4)
        q, k, v = (torch.randn(len(event), 2, 8, dtype=torch.float64) for _ in range(3))
        torch.testing.assert_close(padded_attention(q, k, v, seq), reference_attention(q, k, v, seq))

    @unittest.skipUnless(HAS_FLASH, "needs CUDA and torch.nn.attention.varlen")
    def test_flash_attention_matches_exact_attention_forward_and_backward(self):
        lengths = torch.tensor([3, 129, 340, 1, 200])
        seq = SeqInfo.from_event(torch.repeat_interleave(torch.arange(5), lengths).cuda(), 5)
        q, k, v = (torch.randn(int(lengths.sum()), 6, 64, device="cuda", requires_grad=True) for _ in range(3))
        grad = torch.randn_like(q)
        flash = varlen_attention(q.bfloat16(), k.bfloat16(), v.bfloat16(), seq)
        flash_grads = torch.autograd.grad(flash.float(), (q, k, v), grad)
        exact = reference_attention(q, k, v, seq)
        exact_grads = torch.autograd.grad(exact, (q, k, v), grad)
        def relative_error(a, b):
            return float((a.float() - b).norm() / b.norm())
        self.assertLess(relative_error(flash, exact), 1e-2)
        for a, b in zip(flash_grads, exact_grads):
            self.assertLess(relative_error(a, b), 2e-2)

    # ------------------------------------------------------------ masking
    def test_block_masking_invariants(self):
        masker = MultiBlockMasker(num_blocks=4, block_scale=(0.15, 0.2), global_mask_prob=0.25)
        batch = synthetic_batch(seed=4, batch_size=64)
        batch["fcal_coords"][batch["fcal_event"] == 0] = torch.tensor([3, 3, 3], dtype=torch.int32)
        tokens = FlashViT(metadata=None, **SMALL).embed(batch)
        fractions = []
        for _ in range(5):
            target = masker(tokens)
            for b in range(64):
                for kind in (FCAL, AHCAL):
                    selected = (tokens.event == b) & (tokens.kind == kind)
                    n, masked = int(selected.sum()), int((target & selected).sum())
                    if n <= 1:
                        self.assertEqual(masked, 0)
                    else:
                        self.assertGreaterEqual(masked, 1)
                        self.assertLess(masked, n)
                    if kind == FCAL and n >= 8:
                        fractions.append(masked / n)
        self.assertGreater(np.mean(fractions), 0.3)
        self.assertLess(np.mean(fractions), 0.75)
        many = TokenSet(tokens.x, tokens.event, tokens.kind, tokens.pos, tokens.coords, tokens.batch_size)
        rates = torch.stack([masker(many)[tokens.kind == ECAL].float().mean() for _ in range(200)])
        self.assertAlmostEqual(float(rates.mean()), 0.25, delta=0.03)

    def test_a_block_is_the_nearest_neighbourhood_of_its_centre(self):
        masker = MultiBlockMasker(num_blocks=1, block_scale=(0.3, 0.3), block_aspect=(1.0, 1.0))
        coords = torch.stack(torch.meshgrid(torch.arange(4), torch.arange(4), torch.arange(20), indexing="ij"), -1)
        coords = coords.reshape(-1, 3)
        event = torch.zeros(len(coords), dtype=torch.long)
        target = masker._blocks(coords, event, 1)
        self.assertEqual(int(target.sum()), math.ceil(0.3 * len(coords)))
        # Some occupied patch c has every masked patch at least as close as every visible one.
        distance = torch.cdist(coords.float(), coords.float())
        is_ball = (distance[:, target].max(dim=1).values
                   <= distance[:, ~target].min(dim=1).values + 1e-6)
        self.assertTrue(bool(is_ball.any()))

    # ------------------------------------------------------------ JEPA
    def test_predictions_do_not_see_target_content(self):
        jepa = FlashJEPA(**SMALL, predictor_embed_dim=64, predictor_depth=2, predictor_num_heads=4).eval()
        batch = synthetic_batch(seed=5, batch_size=4)
        tokens = jepa.context_encoder.embed(batch)
        mask = jepa.masker(tokens)
        with patch.object(jepa.masker, "forward", return_value=mask):
            first = jepa(batch)
            # Change the deposits inside every masked FASERCal patch.
            fcal = tokens.kind == FCAL
            masked_patches = set(zip(tokens.event[mask & fcal].tolist(), tokens.pos[mask & fcal].tolist()))
            coords = batch["fcal_coords"].long() // torch.tensor([12, 12, 10])
            pos = (coords[:, 0] * 4 + coords[:, 1]) * 20 + coords[:, 2]
            hit = torch.tensor([(int(e), int(p)) in masked_patches
                                for e, p in zip(batch["fcal_event"], pos)])
            self.assertTrue(bool(hit.any()))
            changed = dict(batch, fcal_feats=batch["fcal_feats"] + 5.0 * hit.unsqueeze(1).float())
            second = jepa(changed)
        torch.testing.assert_close(first["prediction"], second["prediction"])
        self.assertFalse(torch.allclose(first["target"], second["target"]))

    def test_gradients_reach_context_and_predictor_but_not_target(self):
        jepa = FlashJEPA(**SMALL, predictor_embed_dim=64, predictor_depth=1, predictor_num_heads=4)
        jepa.train()
        out = jepa(synthetic_batch(seed=6, batch_size=8, n_tracks=1))
        out["loss"].backward()
        for name, parameter in jepa.named_parameters():
            if name.startswith("target_encoder."):
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
            else:
                self.assertIsNotNone(parameter.grad, name)

    def test_target_branch_has_no_dropout_and_ema_update_is_exact(self):
        jepa = FlashJEPA(**SMALL, drop_rate=0.5, drop_path_rate=0.5,
                         predictor_embed_dim=64, predictor_depth=1, predictor_num_heads=4).train()
        batch = synthetic_batch(seed=7, batch_size=3)
        _, _, first = jepa._targets(batch)
        _, _, second = jepa._targets(batch)
        torch.testing.assert_close(first, second)
        self.assertTrue(jepa.target_encoder.training)

        before = {n: p.clone() for n, p in jepa.target_encoder.named_parameters()}
        with torch.no_grad():
            for p in jepa.context_encoder.parameters():
                p.add_(1.0)
        jepa.update_target_encoder(0.9)
        context = dict(jepa.context_encoder.named_parameters())
        for name, parameter in jepa.target_encoder.named_parameters():
            torch.testing.assert_close(parameter, 0.9 * before[name] + 0.1 * context[name])

    @unittest.skipUnless(HAS_FLASH, "needs CUDA and torch.nn.attention.varlen")
    def test_bf16_flash_training_step_matches_fp32_reference(self):
        jepa = FlashJEPA(**SMALL, predictor_embed_dim=64, predictor_depth=2, predictor_num_heads=4).cuda()
        batch = move_obj(synthetic_batch(seed=8, batch_size=8), "cuda")
        tokens = jepa.context_encoder.embed(batch)
        mask = jepa.masker(tokens)
        with patch.object(jepa.masker, "forward", return_value=mask):
            exact = jepa(batch)["loss"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                flash = jepa(batch)["loss"]
        self.assertLess(abs(float(flash) - float(exact)), 2e-2 * float(exact))

    # ------------------------------------------------------------ transfer
    def test_downstream_model_loads_the_full_target_encoder(self):
        jepa = FlashJEPA(**SMALL, predictor_embed_dim=64, predictor_depth=1, predictor_num_heads=4)
        with torch.no_grad():
            for p in jepa.target_encoder.parameters():
                p.add_(torch.randn_like(p))
        checkpoint = {"state_dict": {f"model.{k}": v for k, v in jepa.state_dict().items()},
                      "flash_encoder_config": jepa.encoder_config}
        downstream = FlashViT(metadata=None, **SMALL)
        self.assertEqual(downstream.encoder_config, checkpoint["flash_encoder_config"])
        report = load_mae_encoder(downstream, checkpoint, encoder_source="auto", require_full_encoder=True)
        self.assertEqual(report["source"], "target")
        self.assertEqual(report["encoder_coverage"], 1.0)
        batch = synthetic_batch(seed=9, batch_size=3)
        jepa.eval(); downstream.eval()
        with torch.no_grad():
            tokens = jepa.target_encoder.embed(batch)
            expected = jepa.target_encoder.forward_tokens(tokens.x, tokens.seq())
            actual, _ = downstream.forward_features(batch)
        torch.testing.assert_close(actual, expected)

    def test_layer_decay_groups_follow_encoder_depth(self):
        from utils.lr_decay import param_groups_lrd

        model = FlashViT(metadata=None, embed_dim=96, depth=4, num_heads=4)
        groups = param_groups_lrd(model, weight_decay=0.05, layer_decay=0.5,
                                  no_weight_decay_list=model.no_weight_decay())
        scale = {}
        for group in groups:
            for parameter in group["params"]:
                scale[id(parameter)] = group["lr_scale"]
        named = dict(model.named_parameters())
        self.assertEqual(scale[id(named["fcal_patch_embed.proj1.weight"])], 0.5 ** 6)
        self.assertEqual(scale[id(named["token_type_embed.weight"])], 0.5 ** 6)
        self.assertEqual(scale[id(named["blocks.0.attn.qkv.weight"])], 0.5 ** 5)
        self.assertEqual(scale[id(named["blocks.3.attn.qkv.weight"])], 0.5 ** 2)
        self.assertEqual(scale[id(named["heads.flavour.2.weight"])], 1.0)

    @unittest.skipUnless(METADATA.is_file(), "local metadata are not available")
    def test_finetuner_step_uses_the_packed_batch(self):
        import pickle
        from model.lightning_model_flash import FlashViTFineTuner

        with METADATA.open("rb") as handle:
            metadata = pickle.load(handle)
        model = FlashViT(metadata=metadata, **SMALL)
        args = SimpleNamespace(
            warmup_epochs=1, cosine_annealing_epochs=1, lr=1e-4, blr=None, batch_size=4,
            layer_decay=0.75, ema_decay=0.9999, beta1=0.9, beta2=0.999, weight_decay=0.05, eps=1e-8,
        )
        module = FlashViTFineTuner(model=model, args=args)
        batch = synthetic_batch(seed=10, batch_size=4)
        loss, parts, _, _, batch_size = module.common_step(batch)
        self.assertEqual(batch_size, 4)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.blocks[0].attn.qkv.weight.grad)
        checkpoint = {}
        module.on_save_checkpoint(checkpoint)
        self.assertEqual(checkpoint["flash_encoder_config"], model.encoder_config)

    def test_pretrainer_optimises_only_context_and_predictor(self):
        from model.lightning_model_flash import param_groups_weight_decay

        jepa = FlashJEPA(**SMALL, predictor_embed_dim=64, predictor_depth=1, predictor_num_heads=4)
        groups = param_groups_weight_decay(jepa, 0.05, jepa.no_weight_decay())
        optimised = {id(p) for group in groups for p in group["params"]}
        names = {id(p): n for n, p in jepa.named_parameters()}
        self.assertFalse(any(names[i].startswith("target_encoder.") for i in optimised))
        no_decay = {names[id(p)] for p in groups[0]["params"]}
        self.assertIn("context_encoder.token_type_embed.weight", no_decay)
        self.assertIn("predictor.mask_token", no_decay)
        self.assertNotIn("context_encoder.blocks.0.attn.qkv.weight", no_decay)

    # ------------------------------------------------------------ data / eval glue
    def test_collate_targets_match_the_spconv_collate(self):
        from utils.funcs import SparseConvTensor, collate

        if SparseConvTensor is None:
            self.skipTest("spconv is not installed")
        rng = np.random.default_rng(11)
        events = [synthetic_event(rng) for _ in range(4)]
        events.insert(2, synthetic_event(rng, n_fcal=0))
        flash, reference = collate_tokens(events, test=False), collate(events, test=False)
        for key in ("flavour_label", "charm_label", "vis_sp_momentum", "jet_momentum", "primary_vertex", "is_cc"):
            torch.testing.assert_close(flash[key], reference[key])
        self.assertEqual(flash["batch_size"], reference["batch_size"])
        torch.testing.assert_close(flash["ecal_hits"], reference["ecal_hits"])

    def test_evaluation_builds_flash_models_from_checkpoints(self):
        from evaluation.inference_common import _arrange_batch, build_eval_args, load_models_from_paths

        args = build_eval_args("unused", "unused", model="tiny", backbone="flash")
        self.assertEqual(args.model, "tiny")
        model = FlashViT(metadata=None, **SMALL)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "epoch=3-loss_vertex_val=0.500000.ckpt"
            torch.save({"epoch": 3, "flash_encoder_config": model.encoder_config,
                        "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, path)
            models = load_models_from_paths({"vertex": str(path)}, args, None, torch.device("cpu"), verbose=False)
        batch = synthetic_batch(seed=12, batch_size=3)
        batch_input, batch_global, target = _arrange_batch(batch, torch.device("cpu"))
        self.assertIsInstance(batch_input, EventBatch)
        with torch.no_grad():
            torch.testing.assert_close(
                models["vertex"](batch_input, batch_global)["out_vertex"],
                model.eval()(EventBatch(batch))["out_vertex"])

    def test_provenance_requires_matching_backbone(self):
        from evaluation.create_csv_v7_ft import read_hparams, validate_pretraining_provenance

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_dir = root / "ft"
            checkpoint_dir.mkdir()
            hparams_path = root / "hparams.yaml"
            pretraining = root / "pretrain.ckpt"
            jepa = flash_jepa_tiny(embed_dim=96, depth=1, num_heads=4, predictor_depth=1,
                                   predictor_embed_dim=64, predictor_num_heads=4)
            state = {f"model.{k}": v for k, v in jepa.state_dict().items()}
            torch.save({"state_dict": state, "flash_encoder_config": jepa.encoder_config}, pretraining)
            for backbone, ok in (("flash", True), ("spconv", False)):
                hparams_path.write_text(
                    f"model: tiny\nbackbone: {backbone}\npretrained_encoder_resolved_source: target\n"
                    f"pretrained_encoder_coverage: 1.0\nload_checkpoint: {pretraining}\n"
                    f"checkpoint_name: ft\ncheckpoint_path: {root}\n")
                hparams = read_hparams(hparams_path)
                if ok:
                    provenance = validate_pretraining_provenance(hparams, hparams_path, checkpoint_dir)
                    self.assertEqual((provenance["backbone"], provenance["model"]), ("flash", "tiny"))
                else:
                    with self.assertRaisesRegex(RuntimeError, "flash-attention checkpoint"):
                        validate_pretraining_provenance(hparams, hparams_path, checkpoint_dir)

    def test_launcher_defaults_match_the_controlled_comparison(self):
        pretrain = (REPO_ROOT / "pretrain_flash.sh").read_text()
        # Effective batch 4,096 = 2 GPUs x 2,048 x accumulation 1.
        for line in ('model="${MODEL:-base}"', 'batch_size="${BATCH_SIZE:-2048}"',
                     'accum_grad_batches="${ACCUM_GRAD_BATCHES:-1}"', 'blr="${BLR:-1e-4}"',
                     'epochs="${EPOCHS:-400}"', 'warmup_epochs="${WARMUP_EPOCHS:-40}"',
                     'save_top_k="${SAVE_TOP_K:-0}"', 'read -r -a gpus <<< "${GPUS:-0 1}"'):
            self.assertIn(line, pretrain)
        finetune = (REPO_ROOT / "finetune_flash.sh").read_text()
        reference = (REPO_ROOT / "finetune.sh").read_text()
        for key in ("EPOCHS", "BLR", "LAYER_DECAY", "DROP_PATH_RATE", "WARMUP_EPOCHS",
                    "COSINE_ANNEALING_EPOCHS", "EMA_DECAY", "HEAD_INIT", "BATCH_SIZE"):
            line = next(l for l in reference.splitlines() if f"${{{key}:-" in l)
            self.assertIn(line, finetune)


if __name__ == "__main__":
    unittest.main()
