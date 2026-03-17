"""
Shared charge preprocessing for transfer-learning datasets.

This implements the single input preprocessing mode used by the FASERCal
pretraining setup in `pretrain.sh`:

  q_scaled = 10 * q
  u = log1p(q_scaled / k)
  z = (u - mu) / sigma

where `(k, mu, sigma)` come from transfer-dataset metadata
`metadata["q_log1p"]`.
"""

from __future__ import annotations

import pickle as pk

import numpy as np


PRETRAIN_CHARGE_METADATA_KEY = "q_log1p"
PRETRAIN_CHARGE_SCALE = 10.0
DEGENERATE_STATS_THRESHOLD = 1e-7


class LogChargePreprocessor:
    """Applies the pretraining-compatible `--preprocessing_input log` transform."""

    def __init__(
        self,
        metadata_path: str | None,
        *,
        metadata_key: str = PRETRAIN_CHARGE_METADATA_KEY,
        scale: float = PRETRAIN_CHARGE_SCALE,
    ):
        self.metadata_path = metadata_path
        self.metadata_key = metadata_key
        self.scale = float(scale)
        self.params = None

        if metadata_path is None:
            return

        with open(metadata_path, "rb") as fd:
            metadata = pk.load(fd)

        if metadata_key not in metadata:
            raise KeyError(
                f"Metadata key '{metadata_key}' not found in {metadata_path}"
            )

        params = metadata[metadata_key]
        if params.get("transform") != "log1p":
            raise ValueError(
                f"Expected '{metadata_key}' to use transform='log1p', "
                f"got {params.get('transform')!r}"
            )

        self._validate_params(params)
        self.params = params

    def _validate_params(self, params: dict) -> None:
        k = float(params.get("k", 0.0))
        sigma = float(params.get("sigma", 0.0))
        orig_max = float(params.get("orig_max", 0.0))
        if (
            orig_max > 0.0
            and k <= DEGENERATE_STATS_THRESHOLD
            and sigma <= DEGENERATE_STATS_THRESHOLD
        ):
            raise ValueError(
                "Degenerate q_log1p metadata detected: both k and sigma collapsed near zero. "
                "Rebuild the transfer charge metadata with finer charge bins."
            )

    @property
    def enabled(self) -> bool:
        return self.params is not None

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Transform charge values; returns float32 and preserves shape."""
        arr = np.asarray(values, dtype=np.float32)
        if self.params is None:
            return arr.astype(np.float32, copy=False)

        arr = np.clip(arr * self.scale, a_min=0.0, a_max=None)
        k = max(float(self.params["k"]), 1e-8)
        mu = float(self.params["mu"])
        sigma = max(float(self.params["sigma"]), 1e-8)

        out = (np.log1p(arr / k) - mu) / sigma
        return out.astype(np.float32, copy=False)
