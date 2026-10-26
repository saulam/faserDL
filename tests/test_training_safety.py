import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import pytorch_lightning as pl
from model.lightning_model_finetune import ExponentialMovingAverage, ViTFineTuner
from utils import resolve_checkpoint_path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _minimal_finetuner_for_ema():
    """Construct only the state needed by the fine-tuning EMA hooks."""
    module = ViTFineTuner.__new__(ViTFineTuner)
    pl.LightningModule.__init__(module)
    module.model = nn.Linear(3, 2)
    module.ema_decay = 0.9
    module.ema = None
    return module


class TrainingSafetyTests(unittest.TestCase):
    def test_explicit_missing_checkpoint_fails_instead_of_starting_fresh(self):
        self.assertIsNone(resolve_checkpoint_path(None, "Resume"))
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "state.ckpt"
            checkpoint.touch()
            self.assertEqual(
                resolve_checkpoint_path(checkpoint, "Resume"), str(checkpoint)
            )
            with self.assertRaisesRegex(FileNotFoundError, "Resume checkpoint"):
                resolve_checkpoint_path(checkpoint.with_name("missing.ckpt"), "Resume")

    def test_finetune_ema_resume_accepts_legacy_checkpoint_and_restores_state(self):
        source = _minimal_finetuner_for_ema()
        source.ema = ExponentialMovingAverage(
            source.model.parameters(), decay=source.ema_decay
        )
        with torch.no_grad():
            for shadow in source.ema.shadow_params:
                shadow.add_(2.0)
        checkpoint = {"ema_state_dict": source.ema.state_dict()}

        restored = _minimal_finetuner_for_ema()
        restored.on_load_checkpoint(checkpoint)
        for expected, actual in zip(
            source.ema.shadow_params, restored.ema.shadow_params
        ):
            torch.testing.assert_close(expected, actual)

        legacy = _minimal_finetuner_for_ema()
        legacy.on_load_checkpoint({})
        self.assertIsNotNone(legacy.ema)

    def test_finetune_ema_is_discarded_on_nonzero_ddp_rank_after_resume(self):
        module = _minimal_finetuner_for_ema()
        module.on_load_checkpoint({})
        module._trainer = SimpleNamespace(is_global_zero=False)
        lightning_optimizer = SimpleNamespace(
            _optimizer=SimpleNamespace(param_groups=[]),
            param_groups=None,
        )
        with patch.object(module, "optimizers", return_value=lightning_optimizer):
            module.on_train_start()
        self.assertIsNone(module.ema)

    def test_launcher_defaults_preserve_paper_backbone_and_jepa_safety(self):
        finetune = (REPO_ROOT / "finetune.sh").read_text()
        scratch = (REPO_ROOT / "scratch.sh").read_text()
        pretrain = (REPO_ROOT / "pretrain.sh").read_text()

        self.assertIn('model="${MODEL:-base}"', finetune)
        self.assertIn('model="${MODEL:-base}"', scratch)
        self.assertIn('seed="${SEED:-7}"', pretrain)
        self.assertIn('--seed $seed', pretrain)
        self.assertIn('save_top_k=0', pretrain)
        self.assertIn('early_stop_patience=0', pretrain)

    def test_jepa_pretraining_skips_only_unused_relational_labels(self):
        try:
            from train.pretrain import skip_unused_jepa_labels
        except (ModuleNotFoundError, NameError):
            self.skipTest("train.pretrain needs the spconv models")

        for objective, expected in (("jepa", False), ("mae", True)):
            datasets = [SimpleNamespace(stage1=True), SimpleNamespace(stage1=True)]
            skip_unused_jepa_labels(SimpleNamespace(pretraining_objective=objective), *datasets)
            self.assertEqual([d.stage1 for d in datasets], [expected, expected])

    @unittest.skipUnless(
        (REPO_ROOT / "events_v7.0_500_npz" / "metadata_stats.pkl").is_file(),
        "local GENIE events are not available",
    )
    def test_skipping_relational_labels_leaves_augmented_inputs_unchanged(self):
        import numpy as np
        from dataset import SparseFASERCALMapDataset

        def make(stage1):
            dataset = SparseFASERCALMapDataset(SimpleNamespace(
                train=True, stage1=stage1, preprocessing_input="log", preprocessing_output="log",
                label_smoothing=0.02, mixup_alpha=0.0,
                dataset_path=str(REPO_ROOT / "events_v7.0_500_npz"),
                metadata_path=str(REPO_ROOT / "events_v7.0_500_npz" / "metadata_stats.pkl"),
            ))
            dataset.augmentations_enabled = True
            return dataset

        with_labels, without_labels = make(True), make(False)
        keys = ("coords", "modules", "feats", "ahcal_hits_coords", "ahcal_hits_feats",
                "ecal_hits", "nb_muspec_tracks", "muspec_info")
        for seed, index in enumerate(range(0, 400, 20)):
            np.random.seed(seed)
            a = with_labels[index]
            np.random.seed(seed)
            b = without_labels[index]
            self.assertIn("csr_hie_ids", a)
            self.assertNotIn("csr_hie_ids", b)
            for key in keys:
                torch.testing.assert_close(torch.as_tensor(a[key]), torch.as_tensor(b[key]), rtol=0, atol=0)

    def test_evaluated_checkpoint_monitors_are_exactly_those_the_test_evaluation_uses(self):
        from evaluation.inference_common import TASK_MONITORS
        from utils import checkpoint_monitors

        evaluated = checkpoint_monitors("evaluated")
        self.assertEqual({m.replace("/", "_") for m in evaluated}, set(TASK_MONITORS.values()))
        self.assertEqual(len(checkpoint_monitors("all")), 7)
        self.assertLessEqual(set(evaluated), set(checkpoint_monitors("all")))

    def test_checkpoint_retention_defaults_are_unchanged(self):
        from utils import ini_argparse

        args = ini_argparse({"tiny": object(), "base": object()}).parse_args(["--metadata_path", "unused"])
        self.assertEqual((args.save_last, args.checkpoint_monitors), (1, "all"))
        finetune = (REPO_ROOT / "finetune.sh").read_text()
        self.assertIn('save_last="${SAVE_LAST:-1}"', finetune)
        self.assertIn('checkpoint_monitors="${CHECKPOINT_MONITORS:-all}"', finetune)

    def test_cli_default_backbone_is_the_published_base_model(self):
        from utils import ini_argparse

        factories = {"tiny": object(), "base": object()}
        args = ini_argparse(factories).parse_args(["--metadata_path", "unused"])
        self.assertIs(args.model, factories["base"])


if __name__ == "__main__":
    unittest.main()
