import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

try:  # the spconv models; skipped in spconv-free environments (neutrino-jepa)
    import spconv.pytorch  # noqa: F401
    import timm  # noqa: F401
except ModuleNotFoundError as error:
    raise unittest.SkipTest(f"needs {error.name}")

from model.lightning_model_jepa import JEPAPreTrainer
from model.sparsejepa import SparseJEPA
from tests.test_jepa import DummyEncoder


_DETERMINISTIC_STATE = None


def setUpModule():
    global _DETERMINISTIC_STATE
    _DETERMINISTIC_STATE = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )


def tearDownModule():
    # Trainer(deterministic=True) enables deterministic algorithms process-wide;
    # restore the previous state so later CUDA (cuBLAS) tests are unaffected.
    enabled, warn_only = _DETERMINISTIC_STATE
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


class _ToyJEPAPreTrainer(JEPAPreTrainer):
    """Use the production optimizer/EMA hooks with a dense unit-test batch."""

    def common_step(self, batch):
        inputs = batch[0]
        output = self.model(inputs, (), mask_ratio=self.mask_ratio)
        return output["loss"], {}, len(inputs)


class _StopAfterFirstEpoch(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            trainer.should_stop = True


class _CaptureRestoredState(pl.Callback):
    def __init__(self):
        self.state = None
        self.global_step = None
        self.optimizer_state_entries = None
        self.scheduler_step = None

    def on_train_start(self, trainer, pl_module):
        self.state = {
            name: value.detach().cpu().clone()
            for name, value in pl_module.state_dict().items()
        }
        self.global_step = trainer.global_step
        self.optimizer_state_entries = len(trainer.optimizers[0].state)
        self.scheduler_step = trainer.lr_scheduler_configs[0].scheduler.step_num


def _args():
    return SimpleNamespace(
        mask_ratio=0.75,
        warmup_epochs=1,
        cosine_annealing_epochs=1,
        lr=1e-3,
        blr=None,
        batch_size=2,
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.05,
        eps=1e-8,
        jepa_ema_decay=0.5,
        jepa_ema_end_decay=0.9,
    )


def _trainer(*, callbacks):
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        limit_train_batches=2,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
    )


def _flat_parameters(module):
    return torch.cat([p.detach().reshape(-1).clone() for p in module.parameters()])


class _RecordEMAOrdering(pl.Callback):
    """Snapshot the encoders immediately before and after each optimizer step."""

    def __init__(self):
        self.records = []
        self._target_before = None

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Runs after forward/backward, immediately before optimizer.step().
        self._target_before = _flat_parameters(pl_module.model.target_encoder)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._target_before is None:
            return  # accumulation micro-batch without an optimizer step
        self.records.append((
            self._target_before,
            _flat_parameters(pl_module.model.context_encoder),
            _flat_parameters(pl_module.model.target_encoder),
        ))
        self._target_before = None


class JEPAEMAOrderingTests(unittest.TestCase):
    def test_ema_follows_each_optimizer_step_with_post_step_context(self):
        torch.manual_seed(5)
        loader = DataLoader(TensorDataset(torch.randn(8, 4, 8)), batch_size=2, shuffle=False)
        args = _args()
        args.jepa_ema_decay = args.jepa_ema_end_decay = 0.5
        module = _ToyJEPAPreTrainer(SparseJEPA(DummyEncoder(), predictor_hidden_dim=12), args)
        update_calls = []
        original_update = module.model.update_target_encoder

        def counting_update(decay):
            update_calls.append(decay)
            original_update(decay)

        module.model.update_target_encoder = counting_update
        recorder = _RecordEMAOrdering()
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            limit_train_batches=4,
            accumulate_grad_batches=2,
            num_sanity_val_steps=0,
            callbacks=[recorder],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(module, train_dataloaders=loader)

        self.assertEqual(trainer.global_step, 2)
        self.assertEqual(len(update_calls), 2)
        self.assertEqual(len(recorder.records), 2)
        for target_before, context_after, target_after in recorder.records:
            self.assertFalse(torch.equal(target_after, target_before))
            torch.testing.assert_close(target_after, 0.5 * target_before + 0.5 * context_after)

    def test_cross_event_std_detects_collapse_and_ignores_singleton_slots(self):
        mask = torch.tensor([[True, True, True], [True, True, False]])
        collapsed = torch.ones(2, 3, 4)
        self.assertEqual(float(JEPAPreTrainer._cross_event_std(collapsed, mask)), 0.0)

        diverse = collapsed.clone()
        diverse[1, :2] = -1.0
        diverse[1, 2] = 100.0  # unselected, and slot 2 has only one selected event
        self.assertAlmostEqual(float(JEPAPreTrainer._cross_event_std(diverse, mask)), 1.0, places=6)


class JEPALightningResumeTests(unittest.TestCase):
    def test_resume_restores_both_encoders_predictor_optimizer_scheduler_and_step(self):
        torch.manual_seed(11)
        data = TensorDataset(torch.randn(8, 4, 8))
        loader = DataLoader(data, batch_size=2, shuffle=False)

        first_model = SparseJEPA(DummyEncoder(), predictor_hidden_dim=12)
        first_module = _ToyJEPAPreTrainer(first_model, _args())
        first_trainer = _trainer(callbacks=[_StopAfterFirstEpoch()])
        first_trainer.fit(first_module, train_dataloaders=loader)
        self.assertEqual(first_trainer.global_step, 2)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "resume.ckpt"
            first_trainer.save_checkpoint(checkpoint_path)
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            self.assertEqual(checkpoint["global_step"], 2)
            self.assertEqual(len(checkpoint["optimizer_states"]), 1)
            self.assertEqual(len(checkpoint["lr_schedulers"]), 1)

            expected = {
                name: value.detach().cpu()
                for name, value in checkpoint["state_dict"].items()
            }
            self.assertTrue(any("context_encoder" in name for name in expected))
            self.assertTrue(any("target_encoder" in name for name in expected))
            self.assertTrue(any("predictor" in name for name in expected))

            torch.manual_seed(99)
            restored_model = SparseJEPA(DummyEncoder(), predictor_hidden_dim=12)
            restored_module = _ToyJEPAPreTrainer(restored_model, _args())
            capture = _CaptureRestoredState()
            resumed_trainer = _trainer(callbacks=[capture])
            resumed_trainer.fit(
                restored_module,
                train_dataloaders=loader,
                ckpt_path=str(checkpoint_path),
            )

        self.assertEqual(capture.global_step, 2)
        self.assertGreater(capture.optimizer_state_entries, 0)
        self.assertEqual(capture.scheduler_step, 2)
        for name, expected_value in expected.items():
            torch.testing.assert_close(capture.state[name], expected_value)
        self.assertEqual(resumed_trainer.global_step, 4)


if __name__ == "__main__":
    unittest.main()
