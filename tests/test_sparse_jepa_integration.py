import unittest

import torch
from spconv.pytorch import SparseConvTensor

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
