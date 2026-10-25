import unittest

import torch
from spconv.pytorch import SparseConv3d, SparseConvTensor

from model.sparsejepa import SparseJEPA
from model.sparsemaevit import SparseMAEViT
from utils.funcs import move_obj


def _sparse_tensor(coords, spatial_shape=(4, 4, 4)):
    indices = torch.tensor([[0, *coord] for coord in coords], dtype=torch.int32)
    features = torch.linspace(0.1, 1.0, len(coords)).reshape(-1, 1)
    return SparseConvTensor(
        features=features,
        indices=indices,
        spatial_shape=spatial_shape,
        batch_size=1,
    )


def _small_encoder(build_decoder):
    return SparseMAEViT(
        in_chans=1,
        fcal_size=(4, 4, 4),
        module_depth_voxels=2,
        embed_dim=16,
        fcal_patch_size=(2, 2, 1),
        ahcal_size=(4, 4, 4),
        ahcal_patch_size=(2, 2, 2),
        num_module_cls=1,
        num_ahcal_cls=1,
        depth=1,
        ahcal_depth=1,
        io_depth=1,
        io_decode_depth=1,
        num_heads=4,
        num_modes=(2, 2),
        decoder_embed_dim=16,
        decoder_num_heads=4,
        mlp_ratio=2.0,
        build_decoder=build_decoder,
        metadata={},
    )


def _inputs():
    fcal = _sparse_tensor(
        [
            (0, 0, 0), (0, 0, 1), (0, 2, 0), (0, 2, 1),
            (2, 0, 2), (2, 0, 3), (2, 2, 2), (2, 2, 3),
        ]
    )
    ahcal = _sparse_tensor([(0, 0, 0), (0, 2, 0), (2, 0, 2), (2, 2, 2)])
    ecal = torch.linspace(0.0, 1.0, 25).reshape(1, 5, 5)
    muspec = torch.tensor([[[1.0, 0.1, 0.2, 0.3, 0.4]]])
    muspec_mask = torch.ones(1, 1, dtype=torch.bool)
    muspec_counts = torch.ones(1, 1)
    return fcal, (ahcal, ecal, muspec, muspec_mask, muspec_counts)


class SparseJEPAIntegrationTests(unittest.TestCase):
    def test_masked_patch_values_do_not_leak_into_context_latents(self):
        torch.manual_seed(13)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SparseJEPA(_small_encoder(build_decoder=False), predictor_hidden_dim=24)
        model = model.to(device).eval()
        fcal, globals_ = _inputs()
        fcal = move_obj(fcal, device)
        globals_ = move_obj(globals_, device)

        torch.manual_seed(29)
        original = model.context_encoder.forward_encoder(
            fcal, globals_, mask_ratio=0.5
        )
        mask = original[2]

        indices = fcal.indices.long()
        batch, x, y, z = indices.unbind(dim=1)
        patch_z = z
        module = patch_z // 2
        intra = (x // 2) * 4 + (y // 2) * 2 + (patch_z % 2)
        masked_hits = mask[batch, module, intra]
        self.assertTrue(masked_hits.any())

        changed_features = fcal.features.clone()
        changed_features[masked_hits] += 100.0
        changed = SparseConvTensor(
            features=changed_features,
            indices=fcal.indices,
            spatial_shape=fcal.spatial_shape,
            batch_size=fcal.batch_size,
        )

        torch.manual_seed(29)
        perturbed = model.context_encoder.forward_encoder(
            changed, globals_, mask_ratio=0.5
        )
        torch.testing.assert_close(original[2], perturbed[2])
        torch.testing.assert_close(original[0], perturbed[0], atol=1e-5, rtol=1e-5)

        # The full-view teacher must remain sensitive to those same values;
        # otherwise the invariance above would not establish masking isolation.
        torch.manual_seed(31)
        teacher_original = model.target_encoder.forward_encoder(
            fcal, globals_, mask_ratio=0.0
        )[0]
        torch.manual_seed(31)
        teacher_changed = model.target_encoder.forward_encoder(
            changed, globals_, mask_ratio=0.0
        )[0]
        self.assertGreater(
            float((teacher_original - teacher_changed).abs().max()), 1e-6
        )

    def test_real_sparse_patching_forward_backward_and_shapes(self):
        torch.manual_seed(3)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        context = _small_encoder(build_decoder=False).to(device)
        if device.type == "cpu":
            # Generic CPU spconv 2.3.8 has no convolution backward kernel.  Keep
            # the real sparse forward while testing gradients through all later
            # encoder blocks and the predictor on CPU-only CI.
            context.fcal_patch_embed.requires_grad_(False)
            context.ahcal_patch_embed.requires_grad_(False)
        model = SparseJEPA(context, predictor_hidden_dim=24).to(device)
        fcal, globals_ = _inputs()
        fcal = move_obj(fcal, device)
        globals_ = move_obj(globals_, device)

        output = model(fcal, globals_, mask_ratio=0.5)
        self.assertEqual(output["fcal_patch_keep"].shape, (1, 2, 8))
        self.assertEqual(output["ahcal_patch_keep"].shape, (1, 8))
        self.assertEqual(output["prediction"].shape, (1, 3, 16))
        self.assertGreater(int(output["prediction_mask"].sum()), 0)
        self.assertTrue(torch.isfinite(output["loss"]))

        output["loss"].backward()
        self.assertTrue(any(p.grad is not None for p in model.context_encoder.parameters() if p.requires_grad))
        self.assertTrue(any(p.grad is not None for p in model.predictor.parameters()))
        self.assertTrue(all(p.grad is None for p in model.target_encoder.parameters()))

    def test_downstream_vit_reproduces_pooled_jepa_target_latents(self):
        for target_level in ("slot", "patch"):
            with self.subTest(target_level=target_level):
                self._assert_downstream_transfer(target_level)

    def _assert_downstream_transfer(self, target_level):
        """Transferred weights must give SparseViT the exact representation
        the JEPA target branch was trained to produce (mean of valid latents)."""
        from model import jepa_vit_tiny, vit_tiny
        from utils import load_mae_encoder

        torch.manual_seed(17)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        jepa = jepa_vit_tiny(metadata={}, target_level=target_level)
        with torch.no_grad():
            # Make the branches differ so the test also checks branch selection.
            for parameter in jepa.target_encoder.parameters():
                parameter.add_(0.01 * torch.randn_like(parameter))
        checkpoint = {"state_dict": {f"model.{k}": v for k, v in jepa.state_dict().items()}}
        vit = vit_tiny(
            drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
            head_init=2e-5, metadata={},
        )
        report = load_mae_encoder(vit, checkpoint, encoder_source="target", require_full_encoder=True)
        self.assertEqual(report["source"], "target")
        jepa, vit = jepa.to(device).eval(), vit.to(device).eval()
        if device.type == "cpu":
            # Generic CPU spconv lacks the eval-time fused-bias kernel.
            for module in vit.modules():
                if isinstance(module, SparseConv3d):
                    module.train()

        coords = [(1, 2, 3), (13, 30, 25), (14, 30, 26), (40, 5, 47), (20, 20, 120), (47, 47, 199)]
        fcal = SparseConvTensor(
            torch.linspace(0.2, 2.0, len(coords)).reshape(-1, 1),
            torch.tensor([[0, *c] for c in coords], dtype=torch.int32), (48, 48, 200), 1,
        )
        ahcal_coords = [(0, 0, 0), (7, 8, 12), (17, 17, 39)]
        ahcal = SparseConvTensor(
            torch.tensor([[0.5], [1.0], [1.5]]),
            torch.tensor([[0, *c] for c in ahcal_coords], dtype=torch.int32), (18, 18, 40), 1,
        )
        globals_ = (
            ahcal,
            torch.linspace(0.0, 1.0, 25).reshape(1, 5, 5),
            torch.tensor([[[1.0, 0.1, 0.2, 0.3, 0.4]]]),
            torch.ones(1, 1, dtype=torch.bool),
            torch.ones(1, 1),
        )
        fcal, globals_ = move_obj(fcal, device), move_obj(globals_, device)
        with torch.no_grad():
            jepa._configure_target_sparse_convs(device)
            latents, keep = jepa.target_encoder.forward_encoder(fcal, globals_, mask_ratio=0.0)[:2]
            pooled_target = (latents * keep.unsqueeze(-1)).sum(1) / keep.sum(1, keepdim=True)
            pooled_context_latents, context_keep = jepa.context_encoder.forward_encoder(
                fcal, globals_, mask_ratio=0.0
            )[:2]
            pooled_context = (pooled_context_latents * context_keep.unsqueeze(-1)).sum(1) / context_keep.sum(1, keepdim=True)
            downstream = vit.forward_features(fcal, globals_)
        torch.testing.assert_close(downstream, pooled_target)
        self.assertGreater(float((downstream - pooled_context).abs().max()), 1e-4)

    def _patch_model(self, device):
        model = SparseJEPA(
            _small_encoder(build_decoder=False),
            target_level="patch",
            predictor_embed_dim=16,
            predictor_depth=1,
            predictor_num_heads=4,
        )
        return model.to(device)

    def test_patch_targets_are_full_view_tokens_at_masked_positions(self):
        torch.manual_seed(41)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._patch_model(device).eval()
        fcal, globals_ = move_obj(_inputs(), device)

        torch.manual_seed(7)
        output = model(fcal, globals_, mask_ratio=0.5)
        rand_mask, ah_rand_mask = output["fcal_patch_mask"], output["ahcal_patch_mask"]
        ecal_drop, muon_drop = output["ecal_drop"], output["muon_drop"]
        self.assertGreater(int(rand_mask.sum()), 0)

        with torch.no_grad():
            model._configure_target_sparse_convs(device)
            tokens = model.target_encoder.forward_encoder(
                fcal, globals_, mask_ratio=0.0, return_tokens=True
            )[8]
        expected = []
        for b, m, l in torch.nonzero(rand_mask).tolist():
            position = int((tokens["fcal_ids"][b, m] == l).nonzero())
            expected.append(tokens["fcal"][b, m, position])
        for b, a in torch.nonzero(ah_rand_mask).tolist():
            position = int((tokens["ahcal_ids"][b] == a).nonzero())
            expected.append(tokens["ahcal"][b, position])
        expected += [tokens["ecal"][b] for b in torch.nonzero(ecal_drop).flatten().tolist()]
        expected += [tokens["muon"][b] for b in torch.nonzero(muon_drop).flatten().tolist()]

        torch.testing.assert_close(output["selected_target"], torch.stack(expected))
        self.assertEqual(output["selected_prediction"].shape, output["selected_target"].shape)
        counts = [int(rand_mask.sum()), int(ah_rand_mask.sum()), int(ecal_drop.sum()), int(muon_drop.sum())]
        self.assertEqual(torch.bincount(output["target_groups"], minlength=4).tolist(), counts)
        self.assertTrue(torch.isfinite(output["loss"]))

    def test_patch_predictions_do_not_depend_on_masked_content(self):
        torch.manual_seed(43)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._patch_model(device).eval()
        fcal, globals_ = move_obj(_inputs(), device)

        torch.manual_seed(29)
        original = model(fcal, globals_, mask_ratio=0.5)
        b, x, y, z = fcal.indices.long().unbind(dim=1)
        masked_hits = original["fcal_patch_mask"][b, z // 2, (x // 2) * 4 + (y // 2) * 2 + (z % 2)]
        self.assertTrue(masked_hits.any())
        features = fcal.features.clone()
        features[masked_hits] += 100.0
        changed = SparseConvTensor(features, fcal.indices, fcal.spatial_shape, fcal.batch_size)

        torch.manual_seed(29)
        perturbed = model(changed, globals_, mask_ratio=0.5)
        torch.testing.assert_close(original["fcal_patch_mask"], perturbed["fcal_patch_mask"])
        torch.testing.assert_close(
            original["selected_prediction"], perturbed["selected_prediction"], atol=1e-5, rtol=1e-5
        )
        number_fcal = int(original["fcal_patch_mask"].sum())
        self.assertGreater(
            float((original["selected_target"][:number_fcal] - perturbed["selected_target"][:number_fcal]).abs().max()),
            1e-4,
        )

    def test_patch_predictor_queries_are_position_dependent(self):
        torch.manual_seed(47)
        predictor = self._patch_model(torch.device("cpu")).predictor.eval()
        latents = torch.randn(1, 3, 16)
        keep = torch.ones(1, 3, dtype=torch.bool)
        one = torch.zeros(1, dtype=torch.long)
        empty = torch.zeros(0, dtype=torch.long)
        fcal_ids = (torch.zeros(3, dtype=torch.long), torch.tensor([0, 0, 1]), torch.tensor([2, 2, 2]))
        out = predictor(latents, keep, fcal_ids, (empty, empty), one, empty)
        self.assertEqual(out.shape, (4, 16))
        torch.testing.assert_close(out[0], out[1])                         # same patch position
        self.assertGreater(float((out[0] - out[2]).abs().max()), 1e-4)     # other module
        self.assertGreater(float((out[0] - out[3]).abs().max()), 1e-4)     # ECAL query

    def test_patch_gradient_routing(self):
        torch.manual_seed(53)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._patch_model(device)
        if device.type == "cpu":
            model.context_encoder.fcal_patch_embed.requires_grad_(False)
            model.context_encoder.ahcal_patch_embed.requires_grad_(False)
        fcal, globals_ = move_obj(_inputs(), device)
        model(fcal, globals_, mask_ratio=0.5)["loss"].backward()

        self.assertTrue(all(p.grad is None for p in model.target_encoder.parameters()))
        trainable = {n: p for n, p in model.predictor.named_parameters() if p.requires_grad}
        self.assertNotIn("fcal_pos_embed.weight", trainable)               # fixed sin-cos
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable.values()))
        self.assertTrue(any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.context_encoder.parameters() if p.requires_grad
        ))
        names = {n for n, _ in model.named_parameters()}
        self.assertTrue(model.no_weight_decay() <= names)

    def test_patch_level_rejects_slot_only_target_selection(self):
        with self.assertRaisesRegex(ValueError, "slot targets"):
            SparseJEPA(_small_encoder(build_decoder=False), target_level="patch", target_selection="all")

    def test_lightning_common_step_metrics_for_both_target_levels(self):
        from types import SimpleNamespace
        from model.lightning_model_jepa import JEPAPreTrainer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fcal, (ahcal, ecal, muspec, muspec_mask, muspec_counts) = move_obj(_inputs(), device)
        batch = {
            "x_sp": fcal, "ahcal_x_sp": ahcal, "ecal_hits": ecal,
            "muspec_feats": muspec, "muspec_attn_mask": muspec_mask, "muspec_counts": muspec_counts,
        }
        args = SimpleNamespace(
            mask_ratio=0.5, warmup_epochs=0, cosine_annealing_epochs=0, lr=1e-3, blr=None,
            batch_size=1, beta1=0.9, beta2=0.95, weight_decay=0.05, eps=1e-8,
            jepa_ema_decay=0.996, jepa_ema_end_decay=1.0,
        )
        for target_level in ("slot", "patch"):
            with self.subTest(target_level=target_level):
                torch.manual_seed(59)
                model = SparseJEPA(
                    _small_encoder(build_decoder=False), target_level=target_level,
                    predictor_hidden_dim=24, predictor_embed_dim=16, predictor_depth=1, predictor_num_heads=4,
                ).to(device).eval()
                loss, metrics, batch_size = JEPAPreTrainer(model, args).common_step(batch)
                self.assertEqual(batch_size, 1)
                self.assertTrue(torch.isfinite(loss))
                for name in ("jepa/cosine", "jepa/target_event_std", "jepa/context_event_std",
                             "jepa/targets_per_event", "jepa/target_coverage"):
                    self.assertTrue(torch.isfinite(torch.as_tensor(metrics[name])), name)
                self.assertEqual("jepa/loss_fcal" in metrics, target_level == "patch")

    def test_mae_decoder_path_remains_available(self):
        torch.manual_seed(5)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _small_encoder(build_decoder=True).to(device)
        fcal, globals_ = _inputs()
        fcal = move_obj(fcal, device)
        globals_ = move_obj(globals_, device)
        output = model(
            fcal,
            globals_,
            mask_ratio=0.5,
            do_relational=False,
        )
        predictions = output[0]
        self.assertIn("occ", predictions)
        self.assertIn("reg", predictions)
        self.assertIn("occ_ah", predictions)
        self.assertIn("reg_ah", predictions)
        self.assertEqual(predictions["occ"].shape[-1], 4)
        self.assertEqual(predictions["occ_ah"].shape[-1], 8)


if __name__ == "__main__":
    unittest.main()
