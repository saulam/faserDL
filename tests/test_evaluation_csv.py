import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from evaluation.create_csv_v7_ft import (
    completed_validation_epochs,
    configure_multiprocessing,
    deterministic_split_files,
    discover_hparams,
    event_identity,
    file_list_fingerprint,
    read_hparams,
    require_checkpoints_match_metrics,
    require_same_inventory,
    validate_jepa_provenance,
    validate_results_frame,
)
from evaluation.inference_common import (
    RESULT_COLUMNS,
    best_checkpoint,
    checkpoint_metadata,
    extract_finetuned_model_state,
    resolve_task_checkpoints,
    run_inference,
)


class EvaluationCSVTests(unittest.TestCase):
    def test_configures_file_system_tensor_sharing(self):
        with patch(
            "evaluation.create_csv_v7_ft.torch.multiprocessing.set_sharing_strategy"
        ) as set_strategy:
            configure_multiprocessing()

        set_strategy.assert_called_once_with("file_system")

    def test_best_checkpoint_is_lowest_metric_and_resolves_all_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            monitors = {
                "flavour": "loss_cls_flavour_val",
                "vertex": "loss_vertex_val",
            }
            for monitor in monitors.values():
                directory = base / monitor
                directory.mkdir()
                (directory / "last.ckpt").touch()
                (directory / f"epoch=2-{monitor}=0.500000.ckpt").touch()
                (directory / f"epoch=4-{monitor}=0.250000.ckpt").touch()

            selected = resolve_task_checkpoints(str(base), monitors=monitors)
            for task, path in selected.items():
                self.assertIn("epoch=4", path, task)
                info = checkpoint_metadata(path)
                self.assertEqual(info["epoch"], 4)
                self.assertAlmostEqual(info["value"], 0.25)

            empty = base / "empty"
            empty.mkdir()
            (empty / "last.ckpt").touch()
            with self.assertRaisesRegex(FileNotFoundError, "metric-named"):
                best_checkpoint(str(empty))

    def test_deterministic_test_split_matches_training_algorithm(self):
        files = [f"/data/events/file_{index:02d}.npz" for index in range(20)]
        train, validation, test = deterministic_split_files(files, seed=7)
        self.assertEqual((len(train), len(validation), len(test)), (17, 1, 2))

        expected_subsets = torch.utils.data.random_split(
            range(len(files)),
            [17, 1, 2],
            generator=torch.Generator().manual_seed(7),
        )
        self.assertEqual(
            test,
            [files[index] for index in expected_subsets[2].indices],
        )
        self.assertEqual(test, deterministic_split_files(files, seed=7)[2])
        self.assertNotEqual(test, deterministic_split_files(files, seed=8)[2])

    def test_inventory_identity_ignores_storage_root_but_preserves_order(self):
        left = ["/scratch/a/events_500/one.npz", "/scratch/a/events_tau/two.npz"]
        right = ["/scratch4/b/events_500/one.npz", "/scratch4/b/events_tau/two.npz"]
        require_same_inventory(left, right)
        self.assertEqual(file_list_fingerprint(left), file_list_fingerprint(right))
        self.assertEqual(event_identity(left[0]), "events_500/one.npz")
        with self.assertRaisesRegex(RuntimeError, "first difference"):
            require_same_inventory(left, list(reversed(right)))

    def test_hparams_parser_and_latest_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "logs" / "run"
            for version in (0, 2, 1):
                directory = root / f"version_{version}"
                directory.mkdir(parents=True)
                (directory / "hparams.yaml").write_text(
                    "model: !!python/name:model.sparsevit.vit_tiny ''\n"
                    "pretrained_encoder_coverage: 1.0\n"
                    "load_checkpoint: null\n"
                )
            latest = discover_hparams(root.parent, "run")
            self.assertEqual(latest.parent.name, "version_2")
            parsed = read_hparams(latest)
            self.assertEqual(parsed["model"], "tiny")
            self.assertEqual(parsed["pretrained_encoder_coverage"], 1.0)
            self.assertIsNone(parsed["load_checkpoint"])

    def test_jepa_provenance_accepts_target_and_rejects_scratch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_base = root / "checkpoints"
            checkpoint_dir = checkpoint_base / "fine"
            checkpoint_dir.mkdir(parents=True)
            pretraining = root / "pretrain.ckpt"
            torch.save(
                {
                    "state_dict": {
                        "model.context_encoder.weight": torch.ones(1),
                        "model.target_encoder.weight": torch.ones(1),
                    }
                },
                pretraining,
            )
            hparams = {
                "model": "tiny",
                "pretrained_encoder_resolved_source": "target",
                "pretrained_encoder_coverage": 1.0,
                "load_checkpoint": str(pretraining),
                "checkpoint_path": str(checkpoint_base),
                "checkpoint_name": "fine",
                "pretrained_checkpoint_epoch": 99,
                "pretrained_checkpoint_global_step": 50900,
            }
            report = validate_jepa_provenance(
                hparams, root / "hparams.yaml", checkpoint_dir,
            )
            self.assertEqual(report["model"], "tiny")
            self.assertEqual(report["pretrained_encoder_source"], "target")

            scratch = dict(hparams)
            scratch.update(
                pretrained_encoder_resolved_source=None,
                pretrained_encoder_coverage=None,
                load_checkpoint=None,
            )
            with self.assertRaisesRegex(RuntimeError, "valid pretraining provenance"):
                validate_jepa_provenance(
                    scratch,
                    root / "hparams.yaml",
                    checkpoint_dir,
                    inspect_pretraining_checkpoint=False,
                )

    def test_provenance_accepts_mae_and_rejects_source_checkpoint_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_dir = root / "checkpoints" / "fine"
            checkpoint_dir.mkdir(parents=True)
            jepa = root / "jepa.ckpt"
            torch.save({"state_dict": {
                "model.context_encoder.module_cls_token": torch.ones(1),
                "model.target_encoder.module_cls_token": torch.ones(1),
            }}, jepa)
            mae = root / "mae.ckpt"
            torch.save({"state_dict": {
                "model.module_cls_token": torch.ones(1),
                "model.query_tokens": torch.ones(1),
            }}, mae)

            def hparams(source, checkpoint):
                return {
                    "model": "base",
                    "pretrained_encoder_resolved_source": source,
                    "pretrained_encoder_coverage": 1.0,
                    "load_checkpoint": str(checkpoint),
                    "checkpoint_path": str(root / "checkpoints"),
                    "checkpoint_name": "fine",
                }

            report = validate_jepa_provenance(hparams("mae", mae), root / "h.yaml", checkpoint_dir)
            self.assertEqual(
                (report["pretraining_objective"], report["pretrained_encoder_source"]), ("mae", "mae")
            )
            report = validate_jepa_provenance(hparams("context", jepa), root / "h.yaml", checkpoint_dir)
            self.assertEqual(report["pretraining_objective"], "jepa")
            with self.assertRaisesRegex(RuntimeError, "not a single-encoder MAE checkpoint"):
                validate_jepa_provenance(hparams("mae", jepa), root / "h.yaml", checkpoint_dir)
            with self.assertRaisesRegex(RuntimeError, "not a JEPA checkpoint"):
                validate_jepa_provenance(hparams("target", mae), root / "h.yaml", checkpoint_dir)
            partial = hparams("mae", mae)
            partial["pretrained_encoder_coverage"] = 0.6
            with self.assertRaisesRegex(RuntimeError, "coverage"):
                validate_jepa_provenance(partial, root / "h.yaml", checkpoint_dir)

    def test_completed_validation_epoch_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            metrics = Path(temporary) / "metrics.csv"
            with metrics.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["epoch", "loss_total/val", "step"]
                )
                writer.writeheader()
                writer.writerow({"epoch": 0, "loss_total/val": 2.0, "step": 10})
                writer.writerow({"epoch": 0, "step": 11})
                writer.writerow({"epoch": 1, "loss_total/val": 1.5, "step": 20})
            self.assertEqual(completed_validation_epochs(metrics), {0, 1})

    def test_selected_checkpoints_must_match_chosen_logger_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = root / "metrics.csv"
            fields = [
                "epoch", "loss_cls/flavour/val", "loss_cls/charm/val",
                "loss_vis/geom/val", "loss_jet/geom/val", "loss_vertex/val",
            ]
            with metrics.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "epoch": 3,
                    "loss_cls/flavour/val": 0.25,
                    "loss_cls/charm/val": 0.25,
                    "loss_vis/geom/val": 0.25,
                    "loss_jet/geom/val": 0.25,
                    "loss_vertex/val": 0.25,
                })
            paths = {
                "flavour": str(root / "epoch=3-loss_cls_flavour_val=0.250000.ckpt"),
                "charm": str(root / "epoch=3-loss_cls_charm_val=0.250000.ckpt"),
                "vis_p": str(root / "epoch=3-loss_vis_geom_val=0.250000.ckpt"),
                "jet_p": str(root / "epoch=3-loss_jet_geom_val=0.250000.ckpt"),
                "vertex": str(root / "epoch=3-loss_vertex_val=0.250000.ckpt"),
            }
            require_checkpoints_match_metrics(paths, metrics)
            paths["vertex"] = str(
                root / "epoch=3-loss_vertex_val=0.200000.ckpt"
            )
            with self.assertRaisesRegex(RuntimeError, "do not belong"):
                require_checkpoints_match_metrics(paths, metrics)

    def test_model_state_extraction_excludes_lightning_loss_parameters(self):
        checkpoint = {
            "state_dict": {
                "model.encoder.weight": torch.ones(2),
                "model.head.bias": torch.zeros(1),
                "u_flavour": torch.tensor(1.0),
            }
        }
        state = extract_finetuned_model_state(checkpoint)
        self.assertEqual(set(state), {"encoder.weight", "head.bias"})

    def test_result_schema_and_validation(self):
        row = {column: 0.0 for column in RESULT_COLUMNS}
        row.update(
            run_number=1,
            event_id=2,
            flavour_label_true="nue",
            charm_label_true="no_charm",
            is_cc=True,
            CCnue_prob=1.0,
            nocharm_prob=1.0,
        )
        frame = pd.DataFrame([row], columns=RESULT_COLUMNS)
        validate_results_frame(frame, expected_rows=1)

        duplicated = pd.concat([frame, frame], ignore_index=True)
        with self.assertRaisesRegex(RuntimeError, "duplicated"):
            validate_results_frame(duplicated)

        wrong_order = frame[list(reversed(RESULT_COLUMNS))]
        with self.assertRaisesRegex(RuntimeError, "schema mismatch"):
            validate_results_frame(wrong_order)

    def test_inference_populates_reference_columns_and_derived_quantities(self):
        target = {
            "primary_vertex": torch.tensor([[1.0, 2.0, 3.0]]),
            "flavour_label": torch.tensor([[0]]),
            "charm_label": torch.tensor([[0]]),
            "vis_sp_momentum": torch.tensor([[3.0, 4.0, 0.0]]),
            "jet_momentum": torch.tensor([[1.0, 1.0, 0.0]]),
            "out_lepton_momentum": torch.tensor([[2.0, 3.0, 0.0]]),
            "in_neutrino_pdg": torch.tensor([12]),
            "in_neutrino_energy": torch.tensor([10.0]),
            "is_cc": torch.tensor([True]),
        }

        output = {
            "out_flavour": torch.tensor([[8.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            "out_charm": torch.tensor([[8.0, 0.0, 0.0, 0.0]]),
            "out_vis": {"p_cart": torch.tensor([[3.0, 4.0, 0.0]])},
            "out_jet": {"p_cart": torch.tensor([[1.0, 1.0, 0.0]])},
            "out_vertex": torch.tensor([[1.0, 2.0, 3.0]]),
        }

        class FakeDataset:
            @staticmethod
            def standardize_vertex(value, reverse=False):
                return np.asarray(value)

            @staticmethod
            def decompose_momentum(value):
                value = torch.as_tensor(value)
                magnitude = torch.linalg.norm(value, dim=1)
                direction = value / magnitude[:, None].clamp_min(1e-8)
                return magnitude, direction

        class FakeModel:
            def __call__(self, batch_input, batch_input_global):
                return output

        models = {
            task: FakeModel()
            for task in ("flavour", "charm", "vis_p", "jet_p", "vertex")
        }
        loader = [{"run_number": [501], "event_id": [42]}]
        with patch(
            "evaluation.inference_common._arrange_batch",
            return_value=(None, [], target),
        ):
            results = run_inference(
                models, loader, FakeDataset(), torch.device("cpu"),
            )

        self.assertEqual(tuple(results), RESULT_COLUMNS)
        self.assertEqual(results["run_number"], [501])
        self.assertEqual(results["event_id"], [42])
        self.assertAlmostEqual(results["evis_true"][0], 5.0)
        self.assertAlmostEqual(results["out_lepton_px_reco"][0], 2.0)
        self.assertAlmostEqual(results["out_lepton_py_reco"][0], 3.0)
        self.assertEqual(results["flavour_label_true"], ["nue"])
        self.assertEqual(results["charm_label_true"], ["no_charm"])


if __name__ == "__main__":
    unittest.main()
