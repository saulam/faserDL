import copy
import unittest

import torch
import torch.nn as nn

try:  # the spconv models; skipped in spconv-free environments (neutrino-jepa)
    import spconv.pytorch  # noqa: F401
    import timm  # noqa: F401
except ModuleNotFoundError as error:
    raise unittest.SkipTest(f"needs {error.name}")

from model.sparsejepa import SparseJEPA
from utils.funcs import extract_pretrained_encoder_state, load_mae_encoder, pack_muspec


class DummyEncoder(nn.Module):
    """Small encoder double with the SparseMAEViT JEPA interface."""

    def __init__(self, embed_dim=8):
        super().__init__()
        self.build_decoder = False
        self.num_module_cls = 1
        self.num_ahcal_cls = 1
        self.module_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.slot_bias = nn.Parameter(torch.randn(1, 3, embed_dim) * 0.01)

    def no_weight_decay(self):
        return {"module_cls_token"}

    def forward_encoder(self, x, x_glob, mask_ratio):
        batch_size, _, embed_dim = x.shape
        if mask_ratio > 0:
            visible = x[:, :2].mean(dim=1, keepdim=True)
            rand_mask = torch.tensor(
                [[[False, True], [True, False]]], device=x.device
            ).expand(batch_size, -1, -1)
            ah_rand_mask = torch.tensor(
                [[False, True]], device=x.device
            ).expand(batch_size, -1)
            ecal_drop = torch.ones(batch_size, dtype=torch.bool, device=x.device)
        else:
            visible = x.mean(dim=1, keepdim=True)
            rand_mask = torch.zeros(batch_size, 2, 2, dtype=torch.bool, device=x.device)
            ah_rand_mask = torch.zeros(batch_size, 2, dtype=torch.bool, device=x.device)
            ecal_drop = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        latents = self.projection(visible).expand(-1, 3, -1) + self.slot_bias
        latent_keep = torch.ones(batch_size, 3, dtype=torch.bool, device=x.device)
        fcal_keep = torch.ones(batch_size, 2, 2, dtype=torch.bool, device=x.device)
        ahcal_keep = torch.ones(batch_size, 2, dtype=torch.bool, device=x.device)
        muon_drop = ecal_drop.clone()
        return (
            latents,
            latent_keep,
            rand_mask,
            fcal_keep,
            ahcal_keep,
            ah_rand_mask,
            ecal_drop,
            muon_drop,
        )


class DummyExpandedEncoder(DummyEncoder):
    """Downstream double with one additional, randomly initialized block."""

    def __init__(self, embed_dim=8):
        super().__init__(embed_dim=embed_dim)
        self.extra_block = nn.Linear(embed_dim, embed_dim)


class SparseJEPATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = SparseJEPA(
            DummyEncoder(),
            predictor_hidden_dim=12,
            target_normalization="layernorm",
            loss_type="smooth_l1",
            target_selection="masked_modules",
        )
        self.inputs = torch.randn(2, 4, 8)

    def test_target_initialization_and_gradient_routing(self):
        context_state = self.model.context_encoder.state_dict()
        target_state = self.model.target_encoder.state_dict()
        for name in context_state:
            torch.testing.assert_close(context_state[name], target_state[name])
        self.assertTrue(all(not p.requires_grad for p in self.model.target_encoder.parameters()))

        output = self.model(self.inputs, (), mask_ratio=0.75)
        self.assertEqual(output["prediction"].shape, (2, 3, 8))
        self.assertEqual(output["target"].shape, (2, 3, 8))
        self.assertEqual(output["prediction_mask"].shape, (2, 3))
        self.assertTrue(torch.isfinite(output["loss"]))

        output["loss"].backward()
        self.assertTrue(any(p.grad is not None for p in self.model.context_encoder.parameters() if p.requires_grad))
        self.assertTrue(any(p.grad is not None for p in self.model.predictor.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.target_encoder.parameters()))

    def test_masked_patch_groups_select_corresponding_latent_slots(self):
        output = self.model(self.inputs, (), mask_ratio=0.75)
        expected = torch.ones(2, 3, dtype=torch.bool)
        torch.testing.assert_close(output["prediction_mask"], expected)

    def test_small_optimization_step_and_ema_update(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=1e-3
        )
        before_context = copy.deepcopy(self.model.context_encoder.state_dict())
        before_target = copy.deepcopy(self.model.target_encoder.state_dict())
        loss = self.model(self.inputs, (), mask_ratio=0.75)["loss"]
        loss.backward()
        optimizer.step()

        changed = any(
            not torch.equal(before_context[name], value)
            for name, value in self.model.context_encoder.state_dict().items()
        )
        self.assertTrue(changed)

        self.model.update_target_encoder(decay=0.5)
        for name, target_value in self.model.target_encoder.state_dict().items():
            context_value = self.model.context_encoder.state_dict()[name]
            old_target = before_target[name]
            if target_value.is_floating_point():
                torch.testing.assert_close(
                    target_value,
                    old_target * 0.5 + context_value * 0.5,
                )
            else:
                torch.testing.assert_close(target_value, context_value)

    def test_checkpoint_roundtrip(self):
        checkpoint = {"state_dict": {f"model.{k}": v for k, v in self.model.state_dict().items()}}
        restored = SparseJEPA(DummyEncoder(), predictor_hidden_dim=12)
        stripped = {
            key[len("model."):]: value for key, value in checkpoint["state_dict"].items()
        }
        restored.load_state_dict(stripped, strict=True)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name])

    def test_downstream_loader_prefers_ema_target(self):
        with torch.no_grad():
            for parameter in self.model.context_encoder.parameters():
                parameter.fill_(1.0)
            for parameter in self.model.target_encoder.parameters():
                parameter.fill_(2.0)
        checkpoint = {"state_dict": {f"model.{k}": v for k, v in self.model.state_dict().items()}}

        extracted, source = extract_pretrained_encoder_state(checkpoint, "auto")
        self.assertEqual(source, "target")
        self.assertTrue(extracted)
        self.assertTrue(all(torch.all(value == 2.0) for value in extracted.values()))

        downstream = DummyEncoder()
        report = load_mae_encoder(downstream, checkpoint, encoder_source="auto")
        self.assertEqual(report["source"], "target")
        for parameter in downstream.parameters():
            self.assertTrue(torch.all(parameter == 2.0))

    def test_existing_mae_checkpoint_layout_still_loads(self):
        source = DummyEncoder()
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(3.0)
        checkpoint = {
            "state_dict": {f"model.{key}": value for key, value in source.state_dict().items()}
        }
        downstream = DummyEncoder()
        report = load_mae_encoder(downstream, checkpoint, encoder_source="auto")
        self.assertEqual(report["source"], "mae")
        for parameter in downstream.parameters():
            self.assertTrue(torch.all(parameter == 3.0))

    def test_loader_reports_partial_encoder_coverage(self):
        source = DummyEncoder()
        checkpoint = {
            "state_dict": {f"model.{key}": value for key, value in source.state_dict().items()}
        }
        downstream = DummyExpandedEncoder()
        report = load_mae_encoder(downstream, checkpoint, encoder_source="auto")

        self.assertLess(report["encoder_coverage"], 1.0)
        self.assertEqual(
            report["missing_encoder"],
            ["extra_block.bias", "extra_block.weight"],
        )
        self.assertFalse(report["shape_mismatches"])

        with self.assertRaisesRegex(RuntimeError, "does not fully cover"):
            load_mae_encoder(
                DummyExpandedEncoder(),
                checkpoint,
                encoder_source="auto",
                require_full_encoder=True,
            )

    def test_all_empty_muon_batch_gets_safe_padding_slot(self):
        counts = torch.zeros(2, 1)
        tracks = [torch.empty(0, 5), torch.empty(0, 5)]
        features, mask, returned_counts = pack_muspec(counts, tracks)
        self.assertEqual(features.shape, (2, 1, 5))
        self.assertEqual(mask.shape, (2, 1))
        self.assertFalse(mask.any())
        torch.testing.assert_close(returned_counts, counts)


if __name__ == "__main__":
    unittest.main()
