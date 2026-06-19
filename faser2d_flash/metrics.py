from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch


FLAVOUR_NAMES = ("CC_nue", "CC_numu", "CC_nutau_e", "CC_nutau_mu", "CC_nutau_had", "NC")
CHARM_NAMES = ("no_charm", "charm_e", "charm_mu", "charm_had")


def _classification(confusion: np.ndarray, names: tuple[str, ...]) -> dict[str, Any]:
    true_count = confusion.sum(axis=1)
    pred_count = confusion.sum(axis=0)
    diagonal = np.diag(confusion)
    precision = np.divide(
        diagonal, pred_count, out=np.zeros_like(diagonal, dtype=float), where=pred_count > 0
    )
    recall = np.divide(
        diagonal, true_count, out=np.zeros_like(diagonal, dtype=float), where=true_count > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    total = confusion.sum()
    return {
        "accuracy": float(diagonal.sum() / total) if total else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            name: {
                "support": int(true_count[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, name in enumerate(names)
        },
        "confusion_matrix": confusion.astype(int).tolist(),
    }


def _vector_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    relative_floor: float,
) -> dict[str, Any]:
    error = prediction - truth
    absolute = np.abs(error)
    component_rmse = np.sqrt(np.mean(error**2, axis=0))
    distance = np.linalg.norm(error, axis=1)
    pred_mag = np.linalg.norm(prediction, axis=1)
    true_mag = np.linalg.norm(truth, axis=1)
    magnitude_error = pred_mag - true_mag
    relative = np.abs(magnitude_error) / np.maximum(true_mag, relative_floor)
    dot = np.sum(prediction * truth, axis=1)
    denom = np.maximum(pred_mag * true_mag, 1e-8)
    angle = np.degrees(np.arccos(np.clip(dot / denom, -1.0, 1.0)))
    angle = angle[(pred_mag > 1e-8) & (true_mag > 1e-8)]
    return {
        "component_mae": absolute.mean(axis=0).tolist(),
        "component_rmse": component_rmse.tolist(),
        "vector_l2_mean": float(distance.mean()),
        "vector_l2_median": float(np.median(distance)),
        "magnitude_mae": float(np.abs(magnitude_error).mean()),
        "magnitude_relative_mae": float(relative.mean()),
        "angle_degrees_mean": float(angle.mean()) if len(angle) else 0.0,
        "angle_degrees_median": float(np.median(angle)) if len(angle) else 0.0,
    }


def _position_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    error = prediction - truth
    distance = np.linalg.norm(error, axis=1)
    return {
        "component_mae": np.abs(error).mean(axis=0).tolist(),
        "component_rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "distance_mean": float(distance.mean()),
        "distance_median": float(np.median(distance)),
        "distance_p95": float(np.quantile(distance, 0.95)),
    }


class EvaluationAccumulator:
    def __init__(self, metadata: dict[str, Any]):
        self.metadata = metadata
        self.flavour_confusion = np.zeros((6, 6), dtype=np.int64)
        self.charm_confusion = np.zeros((4, 4), dtype=np.int64)
        self.values: dict[str, list[np.ndarray]] = defaultdict(list)
        self.loss_sums: dict[str, float] = defaultdict(float)
        self.loss_count = 0

    def update(
        self,
        outputs: dict[str, Any],
        targets: dict[str, torch.Tensor],
        loss_metrics: dict[str, torch.Tensor] | None = None,
    ) -> None:
        flavour_true = targets["flavour_label"].detach().cpu().numpy()
        charm_true = targets["charm_label"].detach().cpu().numpy()
        flavour_pred = outputs["out_flavour"].argmax(-1).detach().cpu().numpy()
        charm_pred = outputs["out_charm"].argmax(-1).detach().cpu().numpy()
        np.add.at(self.flavour_confusion, (flavour_true, flavour_pred), 1)
        np.add.at(self.charm_confusion, (charm_true, charm_pred), 1)

        vis_pred = outputs["out_vis"]["p_cart"].detach().float().cpu().numpy()
        jet_pred = outputs["out_jet"]["p_cart"].detach().float().cpu().numpy()
        vertex_pred = outputs["out_vertex"].detach().float().cpu().numpy()
        vertex_stats = self.metadata["target_stats"]["primary_vertex"]
        vertex_pred = vertex_pred * np.asarray(vertex_stats["std"]) + np.asarray(
            vertex_stats["mean"]
        )
        vertex_true = targets["primary_vertex"].detach().float().cpu().numpy()
        vertex_true = vertex_true * np.asarray(vertex_stats["std"]) + np.asarray(
            vertex_stats["mean"]
        )

        for name, value in (
            ("vis_pred", vis_pred),
            ("vis_true", targets["vis_sp_momentum"].detach().float().cpu().numpy()),
            ("jet_pred", jet_pred),
            ("jet_true", targets["jet_momentum"].detach().float().cpu().numpy()),
            ("lepton_pred", vis_pred - jet_pred),
            (
                "lepton_true",
                targets["out_lepton_momentum"].detach().float().cpu().numpy(),
            ),
            ("vertex_pred", vertex_pred),
            ("vertex_true", vertex_true),
        ):
            self.values[name].append(value)
        if loss_metrics:
            batch_size = len(flavour_true)
            self.loss_count += batch_size
            for name, value in loss_metrics.items():
                if name.startswith("uncertainty/"):
                    continue
                self.loss_sums[name] += float(value.detach().cpu()) * batch_size

    def compute(self) -> dict[str, Any]:
        values = {key: np.concatenate(parts, axis=0) for key, parts in self.values.items()}
        stats = self.metadata["target_stats"]
        result = {
            "classification": {
                "flavour": _classification(self.flavour_confusion, FLAVOUR_NAMES),
                "charm": _classification(self.charm_confusion, CHARM_NAMES),
            },
            "regression": {
                "visible_momentum": _vector_metrics(
                    values["vis_pred"],
                    values["vis_true"],
                    relative_floor=float(stats["vis"]["tau_mag"]),
                ),
                "jet_momentum": _vector_metrics(
                    values["jet_pred"],
                    values["jet_true"],
                    relative_floor=float(stats["jet"]["tau_mag"]),
                ),
                "lepton_momentum": _vector_metrics(
                    values["lepton_pred"],
                    values["lepton_true"],
                    relative_floor=float(stats["lep"]["tau_mag"]),
                ),
                "primary_vertex": _position_metrics(
                    values["vertex_pred"], values["vertex_true"]
                ),
            },
        }
        vis_pred_mag = np.linalg.norm(values["vis_pred"], axis=1)
        vis_true_mag = np.linalg.norm(values["vis_true"], axis=1)
        pt_pred = np.linalg.norm(values["vis_pred"][:, :2], axis=1)
        pt_true = np.linalg.norm(values["vis_true"][:, :2], axis=1)
        result["derived"] = {
            "visible_energy_mae": float(np.abs(vis_pred_mag - vis_true_mag).mean()),
            "missing_pt_mae": float(np.abs(pt_pred - pt_true).mean()),
        }
        if self.loss_count:
            result["losses"] = {
                name: total / self.loss_count for name, total in self.loss_sums.items()
            }
        result["num_events"] = int(len(values["vis_true"]))
        return result


def flatten_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            output.update(flatten_metrics(child, next_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)
    elif isinstance(value, list) and value and all(isinstance(item, (int, float)) for item in value):
        for index, item in enumerate(value):
            output[f"{prefix}[{index}]"] = float(item)
    return output
