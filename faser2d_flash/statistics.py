from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


MAD_NORMALIZATION = 1.482602218505602


def madn(values: np.ndarray, axis=None, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values, axis=axis, keepdims=True)
    deviation = np.median(np.abs(values - median), axis=axis)
    return np.maximum(deviation * MAD_NORMALIZATION, eps)


@dataclass
class RunningMoments:
    count: int
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def empty(cls, dimension: int) -> "RunningMoments":
        return cls(0, np.zeros(dimension, dtype=np.float64), np.zeros(dimension, dtype=np.float64))

    @classmethod
    def from_array(cls, values: np.ndarray) -> "RunningMoments":
        values = np.asarray(values, dtype=np.float64)
        if len(values) == 0:
            return cls.empty(values.shape[1])
        mean = values.mean(axis=0)
        return cls(len(values), mean, ((values - mean) ** 2).sum(axis=0))

    def combine(self, other: "RunningMoments") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean.copy()
            self.m2 = other.m2.copy()
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.mean += delta * other.count / total
        self.m2 += other.m2 + delta**2 * self.count * other.count / total
        self.count = total

    def result(self, eps: float = 1e-6) -> dict[str, Any]:
        variance = self.m2 / max(self.count, 1)
        return {
            "count": self.count,
            "mean": self.mean.tolist(),
            "std": np.maximum(np.sqrt(variance), eps).tolist(),
        }


@dataclass
class VectorStats:
    k_T: float
    mu_uT: float
    sigma_uT: float
    k_Z: float
    mu_uZ: float
    sigma_uZ: float
    s_xyz: tuple[float, float, float]
    s_pT: float
    s_mag: float
    tau_pt: float
    tau_mag: float


def vector_stats(values: np.ndarray, *, clamp_pz: bool) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    px, py, pz = values.T
    if clamp_pz:
        pz = np.maximum(pz, 0.0)
    pt = np.sqrt(px**2 + py**2)
    magnitude = np.sqrt(px**2 + py**2 + pz**2)
    k_t = float(madn(pt))
    ut = np.log1p(pt / k_t)
    k_z = float(madn(pz))
    uz = np.log1p(np.maximum(pz / k_z, -0.999999))
    result = VectorStats(
        k_T=k_t,
        mu_uT=float(np.median(ut)),
        sigma_uT=float(madn(ut)),
        k_Z=k_z,
        mu_uZ=float(np.median(uz)),
        sigma_uZ=float(madn(uz)),
        s_xyz=tuple(float(value) for value in madn(values, axis=0)),
        s_pT=float(madn(pt)),
        s_mag=float(madn(magnitude)),
        tau_pt=float(np.percentile(pt, 5.0)),
        tau_mag=float(np.percentile(magnitude, 5.0)),
    )
    return asdict(result)


def target_statistics(
    visible: np.ndarray,
    jet: np.ndarray,
    lepton: np.ndarray,
    vertices: np.ndarray,
    is_cc: np.ndarray,
) -> dict[str, Any]:
    visible = np.asarray(visible, dtype=np.float64)
    jet = np.asarray(jet, dtype=np.float64)
    lepton = np.asarray(lepton, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    is_cc = np.asarray(is_cc, dtype=bool)
    lepton_cc = lepton[is_cc]
    if len(lepton_cc) == 0:
        lepton_cc = lepton[np.linalg.norm(lepton, axis=1) > 0]
    if len(lepton_cc) == 0:
        raise ValueError("Cannot calculate lepton target statistics without a non-zero lepton")

    output = {
        "vis": vector_stats(visible, clamp_pz=False),
        "jet": vector_stats(jet, clamp_pz=True),
        "lep": vector_stats(lepton_cc, clamp_pz=False),
        "primary_vertex": {
            "transform": "zscore",
            "mean": vertices.mean(axis=0).tolist(),
            "std": np.maximum(vertices.std(axis=0), 1e-8).tolist(),
            "min": vertices.min(axis=0).tolist(),
            "max": vertices.max(axis=0).tolist(),
            "count": len(vertices),
        },
    }
    for group_name, values in (("vis", visible), ("jet", jet), ("lep", lepton)):
        for class_name, mask in (("cc", is_cc), ("nc", ~is_cc)):
            subset = values[mask]
            if len(subset) == 0:
                output[group_name][f"tau_pt_{class_name}"] = output[group_name]["tau_pt"]
                output[group_name][f"tau_mag_{class_name}"] = output[group_name]["tau_mag"]
                continue
            pt = np.linalg.norm(subset[:, :2], axis=1)
            magnitude = np.linalg.norm(subset, axis=1)
            output[group_name][f"tau_pt_{class_name}"] = float(np.percentile(pt, 5.0))
            output[group_name][f"tau_mag_{class_name}"] = float(
                np.percentile(magnitude, 5.0)
            )
    return output
