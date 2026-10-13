"""Shared stage-2 checkpoint loading and inference helpers.

The CSV schema and task-specific checkpoint convention intentionally follow the
published fine-tuning inference scripts.  Fine-tuning EMA weights are not used:
the validation losses which select these checkpoints are measured with the raw
model weights, and the reference inference script loads ``state_dict`` directly.
"""

from __future__ import annotations

import gc
import glob
import os
import re
from collections.abc import Mapping

import numpy as np
import torch
import tqdm
from spconv.pytorch import SparseConvTensor

from model import vit_base, vit_tiny
from utils import arrange_input, arrange_truth, ini_argparse


MODEL_FACTORIES = {"tiny": vit_tiny, "base": vit_base}

TASK_MONITORS = {
    "flavour": "loss_cls_flavour_val",
    "charm": "loss_cls_charm_val",
    "vis_p": "loss_vis_geom_val",
    "jet_p": "loss_jet_geom_val",
    "vertex": "loss_vertex_val",
}

RESULT_COLUMNS = (
    "run_number", "event_id",
    "flavour_label_true",
    "CCnue_prob", "CCnumu_prob", "CCnutau2e_prob",
    "CCnutau2mu_prob", "CCnutau2H_prob", "NC_prob",
    "charm_label_true",
    "nocharm_prob", "charm2e_prob", "charm2mu_prob", "charm2H_prob",
    "evis_true", "evis_reco",
    "ptmiss_true", "ptmiss_reco",
    "vis_px_true", "vis_py_true", "vis_pz_true",
    "vis_px_reco", "vis_py_reco", "vis_pz_reco",
    "out_lepton_px_true", "out_lepton_py_true", "out_lepton_pz_true",
    "out_lepton_px_reco", "out_lepton_py_reco", "out_lepton_pz_reco",
    "jet_px_true", "jet_py_true", "jet_pz_true",
    "jet_px_reco", "jet_py_reco", "jet_pz_reco",
    "primary_vertex_x_true", "primary_vertex_y_true", "primary_vertex_z_true",
    "primary_vertex_x_reco", "primary_vertex_y_reco", "primary_vertex_z_reco",
    "in_neutrino_pdg", "in_neutrino_energy", "is_cc",
)

_CHECKPOINT_RE = re.compile(
    r"^epoch=(?P<epoch>\d+)-(?P<metric>[^=]+)="
    r"(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\.ckpt$"
)


def build_eval_args(
    metadata_path: str,
    dataset_path: str,
    *,
    model: str = "tiny",
    batch_size: int = 128,
    num_workers: int = 16,
    dropout: float = 0.0,
    attn_dropout: float = 0.0,
    drop_path_rate: float = 0.2,
    head_init: float = 2e-5,
    head_dropout_cls: float = 0.0,
    head_dropout_reg: float = 0.0,
    preprocessing_input: str = "log",
    preprocessing_output: str = "log",
    label_smoothing: float = 0.02,
):
    """Construct the repository's normal argument namespace for evaluation."""
    parser = ini_argparse(MODEL_FACTORIES)
    return parser.parse_args(
        [
            "--test",
            "--stage2",
            "--augmentations_disabled",
            "--dataset_path", str(dataset_path),
            "--metadata_path", str(metadata_path),
            "--dropout", str(dropout),
            "--attn_dropout", str(attn_dropout),
            "--drop_path_rate", str(drop_path_rate),
            "--model", model,
            "--batch_size", str(batch_size),
            "--num_workers", str(num_workers),
            "--head_init", str(head_init),
            "--head_dropout_cls", str(head_dropout_cls),
            "--head_dropout_reg", str(head_dropout_reg),
            "--preprocessing_input", preprocessing_input,
            "--preprocessing_output", preprocessing_output,
            "--label_smoothing", str(label_smoothing),
            "--mixup_alpha", "0",
        ]
    )


def checkpoint_metadata(path: str) -> dict[str, object]:
    """Return the monitor value encoded in a fine-tuning checkpoint name."""
    match = _CHECKPOINT_RE.match(os.path.basename(path))
    if match is None:
        return {"epoch": None, "monitor": None, "value": None}
    return {
        "epoch": int(match.group("epoch")),
        "monitor": match.group("metric"),
        "value": float(match.group("value")),
    }


def best_checkpoint(checkpoint_dir: str) -> str:
    """Select the lowest monitored-loss checkpoint, never ``last.ckpt``."""
    candidates = []
    for path in glob.glob(os.path.join(checkpoint_dir, "*.ckpt")):
        metadata = checkpoint_metadata(path)
        if metadata["value"] is not None:
            candidates.append((metadata["value"], path))
    if not candidates:
        raise FileNotFoundError(
            f"No metric-named checkpoint found in {checkpoint_dir}; "
            "the fine-tuning run may not have completed validation yet"
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]


def resolve_task_checkpoints(
    checkpoint_base_path: str,
    monitors: Mapping[str, str] = TASK_MONITORS,
) -> dict[str, str]:
    """Resolve the best independently monitored checkpoint for every task."""
    return {
        task: best_checkpoint(os.path.join(checkpoint_base_path, monitor))
        for task, monitor in monitors.items()
    }


def extract_finetuned_model_state(checkpoint: Mapping) -> dict[str, torch.Tensor]:
    """Extract only the downstream model from a Lightning fine-tune checkpoint."""
    if "state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint has no Lightning 'state_dict'")
    state = {
        key[len("model."):]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    if not state:
        raise RuntimeError("Checkpoint contains no 'model.*' parameters")
    return state


def load_models_from_paths(ckpt_paths, args, metadata, device, verbose=True):
    """Load one strict raw-weight model per downstream task checkpoint."""
    models = {}
    for task, checkpoint_path in ckpt_paths.items():
        if verbose:
            info = checkpoint_metadata(checkpoint_path)
            print(
                f"  {task:8s} -> {checkpoint_path} "
                f"(epoch={info['epoch']}, {info['monitor']}={info['value']})"
            )

        model = args.model(
            drop_rate=args.dropout,
            attn_drop_rate=args.attn_dropout,
            drop_path_rate=args.drop_path_rate,
            head_init=args.head_init,
            head_dropout_cls=args.head_dropout_cls,
            head_dropout_reg=args.head_dropout_reg,
            metadata=metadata,
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        encoded_epoch = checkpoint_metadata(checkpoint_path)["epoch"]
        stored_epoch = checkpoint.get("epoch")
        if stored_epoch != encoded_epoch:
            raise RuntimeError(
                f"[{task}] checkpoint filename says epoch {encoded_epoch}, "
                f"but the checkpoint stores epoch {stored_epoch}"
            )
        state = extract_finetuned_model_state(checkpoint)
        model.load_state_dict(state, strict=True)
        del checkpoint, state

        model = model.to(device)
        model.eval()
        models[task] = model
    return models


def _spconv_to_device(x, device):
    indices = x.indices.to(device)
    if indices.dtype != torch.int32:
        indices = indices.int()
    return SparseConvTensor(
        x.features.to(device), indices, x.spatial_shape, x.batch_size,
    )


def _to_device(x, device):
    if isinstance(x, SparseConvTensor):
        return _spconv_to_device(x, device)
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x


def _arrange_batch(batch, device):
    batch_input, *batch_input_global = arrange_input(batch)
    batch_input = _spconv_to_device(batch_input, device)
    batch_input_global = [_to_device(value, device) for value in batch_input_global]
    target = arrange_truth(batch)
    return batch_input, batch_input_global, target


def label2flavour(label):
    return np.where(label == 0, "nue",
           np.where(label == 1, "numu",
           np.where(label == 2, "nutau->e",
           np.where(label == 3, "nutau->mu",
           np.where(label == 4, "nutau->had", "NC")))))


def label2charm(label):
    return np.where(label == 0, "no_charm",
           np.where(label == 1, "charm->e",
           np.where(label == 2, "charm->mu", "charm->had")))


def to_numpy(value, dtype=None):
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray) and value.dtype != object:
        array = value
    elif isinstance(value, np.ndarray):
        array = to_numpy(value.tolist())
    elif isinstance(value, (list, tuple)):
        array = np.stack([to_numpy(item) for item in value])
    else:
        array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def empty_results():
    return {column: [] for column in RESULT_COLUMNS}


@torch.inference_mode()
def run_inference(models, loader, dataset, device, max_batches=None):
    """Evaluate task-specific checkpoints and return the canonical CSV columns."""
    missing_tasks = set(TASK_MONITORS) - set(models)
    if missing_tasks:
        raise ValueError(f"Missing task models: {sorted(missing_tasks)}")

    results = empty_results()
    total = len(loader) if max_batches is None else min(len(loader), max_batches)
    progress = tqdm.tqdm(enumerate(loader), total=total, ascii=True)
    for batch_index, batch in progress:
        if max_batches is not None and batch_index >= max_batches:
            break
        if device.type == "cuda" and batch_index and batch_index % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        batch_input, batch_input_global, target = _arrange_batch(batch, device)
        outputs = {
            name: model(batch_input, batch_input_global)
            for name, model in models.items()
        }
        batch_size = len(outputs["flavour"]["out_flavour"])

        target_vertex = dataset.standardize_vertex(
            to_numpy(target["primary_vertex"], dtype=np.float32), reverse=True,
        )
        target_flavour = to_numpy(target["flavour_label"])
        target_charm = to_numpy(target["charm_label"])
        target_vis = to_numpy(target["vis_sp_momentum"], dtype=np.float32)
        target_jet = to_numpy(target["jet_momentum"], dtype=np.float32)
        target_lepton = to_numpy(target["out_lepton_momentum"], dtype=np.float32)
        target_evis, _ = dataset.decompose_momentum(target_vis)
        target_evis = to_numpy(target_evis)
        target_ptmiss = np.linalg.norm(target_vis[:, :2], axis=1)
        in_neutrino_pdg = to_numpy(target["in_neutrino_pdg"]).reshape(-1)
        in_neutrino_energy = to_numpy(target["in_neutrino_energy"]).reshape(-1)
        is_cc = to_numpy(target["is_cc"], dtype=bool).reshape(-1)

        output_vertex = dataset.standardize_vertex(
            outputs["vertex"]["out_vertex"].float().cpu().numpy(), reverse=True,
        )
        output_flavour = torch.softmax(
            outputs["flavour"]["out_flavour"].float(), dim=-1,
        ).cpu().numpy()
        output_charm = torch.softmax(
            outputs["charm"]["out_charm"].float(), dim=-1,
        ).cpu().numpy()
        output_vis = outputs["vis_p"]["out_vis"]["p_cart"].float().cpu().numpy()
        output_jet = outputs["jet_p"]["out_jet"]["p_cart"].float().cpu().numpy()
        output_lepton = output_vis - output_jet
        output_evis, _ = dataset.decompose_momentum(output_vis)
        output_evis = to_numpy(output_evis)
        output_ptmiss = np.linalg.norm(output_vis[:, :2], axis=1)

        flavour = np.atleast_1d(label2flavour(target_flavour).squeeze())
        charm = np.atleast_1d(label2charm(target_charm).squeeze())
        run_number = batch["run_number"]
        event_id = batch["event_id"]

        for index in range(batch_size):
            row = {
                "run_number": int(run_number[index]),
                "event_id": int(event_id[index]),
                "flavour_label_true": flavour[index].item(),
                "CCnue_prob": output_flavour[index, 0].item(),
                "CCnumu_prob": output_flavour[index, 1].item(),
                "CCnutau2e_prob": output_flavour[index, 2].item(),
                "CCnutau2mu_prob": output_flavour[index, 3].item(),
                "CCnutau2H_prob": output_flavour[index, 4].item(),
                "NC_prob": output_flavour[index, 5].item(),
                "charm_label_true": charm[index].item(),
                "nocharm_prob": output_charm[index, 0].item(),
                "charm2e_prob": output_charm[index, 1].item(),
                "charm2mu_prob": output_charm[index, 2].item(),
                "charm2H_prob": output_charm[index, 3].item(),
                "evis_true": target_evis[index].item(),
                "evis_reco": output_evis[index].item(),
                "ptmiss_true": target_ptmiss[index].item(),
                "ptmiss_reco": output_ptmiss[index].item(),
                "in_neutrino_pdg": int(in_neutrino_pdg[index]),
                "in_neutrino_energy": float(in_neutrino_energy[index]),
                "is_cc": bool(is_cc[index]),
            }
            for prefix, true_value, reco_value in (
                ("vis", target_vis, output_vis),
                ("out_lepton", target_lepton, output_lepton),
                ("jet", target_jet, output_jet),
            ):
                for axis, axis_index in zip("xyz", range(3)):
                    row[f"{prefix}_p{axis}_true"] = true_value[index, axis_index].item()
                    row[f"{prefix}_p{axis}_reco"] = reco_value[index, axis_index].item()
            for axis, axis_index in zip("xyz", range(3)):
                row[f"primary_vertex_{axis}_true"] = target_vertex[index, axis_index].item()
                row[f"primary_vertex_{axis}_reco"] = output_vertex[index, axis_index].item()
            for column in RESULT_COLUMNS:
                results[column].append(row[column])

        del batch_input, batch_input_global, outputs

    return results
