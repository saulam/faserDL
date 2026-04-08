"""Dataset helpers for scintillator PID transfer."""

import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset

from ..transfer_charge_preprocessing import LogChargePreprocessor


CLASS_NAMES = ["proton", "pion", "muon", "electron"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
SCINTILLATOR_CONTEXT_FEATURE_DIM = 31

VOXEL_SIZE_MM = 10.0
COORD_OFFSET_MM = 995.0
DETECTOR_SHAPE = (200, 200, 200)


def _compute_cluster_features(coords, energy, spatial_shape):
    """Cluster features for the context token."""
    coords = np.asarray(coords, dtype=np.float32)
    energy = np.asarray(energy, dtype=np.float32).reshape(-1)
    spatial_shape = np.asarray(spatial_shape, dtype=np.float32)
    if len(coords) == 0:
        return np.zeros((25,), dtype=np.float32)
    spatial_max = np.maximum(spatial_shape - 1.0, 1.0)

    centroid = coords.mean(axis=0) / spatial_max
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    bbox = (maxs - mins + 1.0) / spatial_shape

    num_vox = float(len(coords))
    sum_energy = float(energy.sum())
    mean_energy = sum_energy / max(num_vox, 1.0)
    std_energy = float(energy.std()) if len(energy) > 1 else 0.0
    bbox_vol = float(np.prod(np.maximum(maxs - mins + 1.0, 1.0)))
    density = num_vox / max(bbox_vol, 1.0)
    crop_edge = ((coords == 0.0) | (coords == spatial_max)).any(axis=1)
    edge_fraction = float(crop_edge.mean()) if len(coords) > 0 else 0.0

    principal_dir = np.zeros(3, dtype=np.float32)
    start_pos = centroid.astype(np.float32, copy=True)
    length = 0.0
    linearity = 0.0
    planarity = 0.0
    charge_asym = 0.0

    rel = coords - coords.mean(axis=0, keepdims=True)
    if len(coords) > 1:
        cov = (rel.T @ rel) / float(max(len(coords) - 1, 1))
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 0.0).astype(np.float32)
        evecs = evecs[:, order].astype(np.float32)

        axis = evecs[:, 0]
        proj = rel @ axis
        lo_idx = int(np.argmin(proj))
        hi_idx = int(np.argmax(proj))
        start_raw = coords[lo_idx].copy()
        end_raw = coords[hi_idx].copy()
        midpoint = 0.5 * (float(proj.max()) + float(proj.min()))
        low_mask = proj <= midpoint
        low_energy = float(energy[low_mask].sum())
        high_energy = float(energy[~low_mask].sum())

        # Fix the PCA sign with the charge profile.
        if (high_energy < (low_energy - 1e-6)) or (
            abs(high_energy - low_energy) <= 1e-6
            and float(energy[hi_idx]) < float(energy[lo_idx])
        ):
            start_raw, end_raw = end_raw, start_raw
            low_energy, high_energy = high_energy, low_energy

        delta = end_raw - start_raw
        delta_norm = float(np.linalg.norm(delta))
        length_norm = float(np.linalg.norm(spatial_max))
        length = delta_norm / max(length_norm, 1e-6)
        if delta_norm > 1e-6:
            principal_dir = (delta / delta_norm).astype(np.float32)
        start_pos = (start_raw / spatial_max).astype(np.float32)

        sqrt_evals = np.sqrt(np.maximum(evals, 0.0))
        linearity = float(
            np.log1p(sqrt_evals[0] / max(float(sqrt_evals[1]), 1e-6))
        )
        planarity = float(
            np.log1p(sqrt_evals[1] / max(float(sqrt_evals[2]), 1e-6))
        )
        charge_asym = (high_energy - low_energy) / max(sum_energy, 1e-6)
    else:
        evals = np.zeros(3, dtype=np.float32)
    pca_scales = np.log1p(np.sqrt(evals))

    return np.asarray(
        [
            centroid[0], centroid[1], centroid[2],
            bbox[0], bbox[1], bbox[2],
            np.log1p(num_vox),
            np.log1p(sum_energy),
            np.log1p(mean_energy),
            np.log1p(std_energy),
            np.log1p(density),
            edge_fraction,
            pca_scales[0], pca_scales[1], pca_scales[2],
            start_pos[0], start_pos[1], start_pos[2],
            principal_dir[0], principal_dir[1], principal_dir[2],
            length,
            linearity,
            planarity,
            charge_asym,
        ],
        dtype=np.float32,
    )


class ScintillatorPIDDataset(Dataset):
    """Single-particle PID dataset from the public scintillator .pt files."""

    def __init__(
        self,
        data_dir: str,
        split: str = "training",
        spatial_shape: tuple[int, int, int] = DETECTOR_SHAPE,
        augment: bool = False,
        charge_metadata_path: str | None = None,
    ):
        self.spatial_shape = tuple(spatial_shape)
        if any(s > d for s, d in zip(self.spatial_shape, DETECTOR_SHAPE)):
            raise ValueError(
                f"spatial_shape {self.spatial_shape} cannot exceed detector shape {DETECTOR_SHAPE}"
            )
        self.augment = augment
        self._can_apply_cube_symmetry = len(set(self.spatial_shape)) == 1
        self.charge_preprocessor = LogChargePreprocessor(charge_metadata_path)

        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.samples = []
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(f"Class directory not found: {class_dir}")
            idx = CLASS_TO_IDX[class_name]
            paths = sorted(glob.glob(os.path.join(class_dir, "event*.pt")))
            self.samples.extend((p, idx) for p in paths)

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found under {split_dir}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _deduplicate(coords, feats):
        """Merge voxels at the same grid cell (sum energy)."""
        _, inv, counts = np.unique(
            coords, axis=0, return_inverse=True, return_counts=True,
        )
        if counts.max() == 1:
            return coords, feats
        n_unique = len(counts)
        new_feats = np.zeros((n_unique, feats.shape[1]), dtype=np.float32)
        np.add.at(new_feats, inv, feats)
        new_coords = np.zeros((n_unique, 3), dtype=coords.dtype)
        new_coords[inv] = coords
        return new_coords, new_feats

    @staticmethod
    def _transform_boundary_features(boundary_features, axis_perm, axis_flip):
        """Express detector-boundary distances in the augmented local frame."""
        if boundary_features is None:
            return None
        axis_perm = np.asarray(axis_perm, dtype=np.int64)
        axis_flip = np.asarray(axis_flip, dtype=bool)
        boundary = boundary_features.reshape(3, 2).copy()
        out_boundary = boundary[axis_perm].copy()
        for out_axis, do_flip in enumerate(axis_flip):
            if do_flip:
                out_boundary[out_axis] = out_boundary[out_axis, ::-1]
        return out_boundary.reshape(-1)

    @staticmethod
    def _sample_cube_symmetry(spatial_shape):
        """Sample a signed axis permutation for cubic local crops."""
        if len(set(spatial_shape)) != 1:
            return np.arange(3, dtype=np.int64), np.zeros((3,), dtype=bool)
        return np.random.permutation(3).astype(np.int64), (np.random.rand(3) < 0.5)

    @staticmethod
    def _apply_cube_symmetry(coords, spatial_shape, axis_perm, axis_flip, boundary_features=None):
        """Apply a cube symmetry to the local crop and detector-context features."""
        axis_perm = np.asarray(axis_perm, dtype=np.int64)
        axis_flip = np.asarray(axis_flip, dtype=bool)
        out = coords[:, axis_perm].copy()
        spatial_max = np.asarray(spatial_shape, dtype=np.int32) - 1
        for out_axis, do_flip in enumerate(axis_flip):
            if do_flip:
                out[:, out_axis] = spatial_max[out_axis] - out[:, out_axis]

        if boundary_features is None:
            return out
        out_boundary = ScintillatorPIDDataset._transform_boundary_features(
            boundary_features, axis_perm, axis_flip,
        )
        return out, out_boundary

    @staticmethod
    def _boundary_features_from_origin(origin, spatial_shape):
        detector_shape = np.array(DETECTOR_SHAPE, dtype=np.int32)
        crop_shape = np.array(spatial_shape, dtype=np.int32)
        max_origin = np.maximum(detector_shape - crop_shape, 1)
        origin_f = origin.astype(np.float32)
        max_origin_f = max_origin.astype(np.float32)
        dist_lo = origin_f / max_origin_f
        dist_hi = (detector_shape - crop_shape - origin).astype(np.float32) / max_origin_f
        boundary_features = np.empty(6, dtype=np.float32)
        boundary_features[0::2] = dist_lo
        boundary_features[1::2] = dist_hi
        return boundary_features

    @staticmethod
    def _random_translate(
        coords,
        spatial_shape,
        origin,
        max_shift=(6, 6, 6),
        axis_perm=None,
        axis_flip=None,
    ):
        """Translate an augmented local crop and keep detector context aligned."""
        coords = np.asarray(coords, dtype=np.int32)
        axis_perm = (
            np.arange(3, dtype=np.int64)
            if axis_perm is None
            else np.asarray(axis_perm, dtype=np.int64)
        )
        axis_flip = (
            np.zeros((3,), dtype=bool)
            if axis_flip is None
            else np.asarray(axis_flip, dtype=bool)
        )
        if coords.size == 0:
            boundary = ScintillatorPIDDataset._boundary_features_from_origin(origin, spatial_shape)
            boundary = ScintillatorPIDDataset._transform_boundary_features(
                boundary, axis_perm, axis_flip,
            )
            return coords, origin.copy(), boundary

        max_shift = np.asarray(max_shift, dtype=np.int32)
        spatial_max = np.asarray(spatial_shape, dtype=np.int32) - 1
        detector_shape = np.asarray(DETECTOR_SHAPE, dtype=np.int32)
        crop_shape = np.asarray(spatial_shape, dtype=np.int32)
        max_origin = detector_shape - crop_shape
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        low = -np.minimum(max_shift, mins)
        high = np.minimum(max_shift, spatial_max - maxs)

        for out_axis, src_axis in enumerate(axis_perm):
            if axis_flip[out_axis]:
                det_low = -origin[src_axis]
                det_high = max_origin[src_axis] - origin[src_axis]
            else:
                det_low = origin[src_axis] - max_origin[src_axis]
                det_high = origin[src_axis]
            low[out_axis] = max(int(low[out_axis]), int(det_low))
            high[out_axis] = min(int(high[out_axis]), int(det_high))

        shift = np.zeros((3,), dtype=np.int32)
        for d in range(3):
            lo = int(low[d])
            hi = int(high[d])
            if lo == hi:
                shift[d] = lo
            else:
                shift[d] = int(np.random.randint(lo, hi + 1))

        out_coords = coords + shift
        origin_delta = np.zeros((3,), dtype=np.int32)
        for out_axis, src_axis in enumerate(axis_perm):
            sign = -1 if axis_flip[out_axis] else 1
            origin_delta[src_axis] = sign * shift[out_axis]
        out_origin = origin - origin_delta
        out_boundary = ScintillatorPIDDataset._boundary_features_from_origin(
            out_origin, spatial_shape,
        )
        out_boundary = ScintillatorPIDDataset._transform_boundary_features(
            out_boundary, axis_perm, axis_flip,
        )
        return out_coords, out_origin, out_boundary

    @staticmethod
    def _compute_crop_origin(coords, spatial_shape):
        crop_shape = np.array(spatial_shape, dtype=np.int32)
        detector_shape = np.array(DETECTOR_SHAPE, dtype=np.int32)
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        bbox_center = 0.5 * (bbox_min + bbox_max)
        half_extent = 0.5 * (crop_shape - 1)
        origin = np.floor(bbox_center - half_extent).astype(np.int32)
        max_origin = detector_shape - crop_shape
        return np.clip(origin, 0, max_origin)

    def _crop_with_context(self, coords, feats):
        """Crop around the particle while encoding where the crop sits in the detector."""
        crop_shape = np.array(self.spatial_shape, dtype=np.int32)
        detector_shape = np.array(DETECTOR_SHAPE, dtype=np.int32)
        if coords.shape[0] == 0:
            origin = np.maximum((detector_shape - crop_shape) // 2, 0)
        else:
            origin = self._compute_crop_origin(coords, self.spatial_shape)
        upper = origin + crop_shape

        keep = ((coords >= origin) & (coords < upper)).all(axis=1)
        local_coords = coords[keep] - origin
        local_feats = feats[keep]
        boundary_features = self._boundary_features_from_origin(origin, self.spatial_shape)
        return local_coords.astype(np.int32), local_feats, origin, boundary_features

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = torch.load(path, weights_only=False)

        reco = data["reco_hits"]
        # Convert mm coordinates to detector voxels.
        coords = np.rint((reco[:, :3] + COORD_OFFSET_MM) / VOXEL_SIZE_MM).astype(np.int32)
        coords = np.clip(coords, 0, np.asarray(DETECTOR_SHAPE, dtype=np.int32) - 1)
        energy = reco[:, 3:4].astype(np.float32)

        coords, energy = self._deduplicate(coords, energy)
        coords, energy, origin, boundary_features = self._crop_with_context(coords, energy)

        if self.augment:
            axis_perm = np.arange(3, dtype=np.int64)
            axis_flip = np.zeros((3,), dtype=bool)
            if self._can_apply_cube_symmetry:
                axis_perm, axis_flip = self._sample_cube_symmetry(self.spatial_shape)
            coords, boundary_features = self._apply_cube_symmetry(
                coords, self.spatial_shape, axis_perm, axis_flip, boundary_features,
            )
            coords, origin, boundary_features = self._random_translate(
                coords,
                self.spatial_shape,
                origin,
                axis_perm=axis_perm,
                axis_flip=axis_flip,
            )

        cluster_features = _compute_cluster_features(coords, energy, self.spatial_shape)
        context_features = np.concatenate(
            [cluster_features, boundary_features.astype(np.float32, copy=False)],
            axis=0,
        )
        if context_features.shape[0] != SCINTILLATOR_CONTEXT_FEATURE_DIM:
            raise RuntimeError(
                "Unexpected scintillator context feature dim "
                f"{context_features.shape[0]} != {SCINTILLATOR_CONTEXT_FEATURE_DIM}"
            )

        energy = self.charge_preprocessor.transform(energy)

        return {
            "coords": torch.from_numpy(coords).int(),
            "feats": torch.from_numpy(energy),
            "label": torch.tensor(label, dtype=torch.long),
            "global_features": torch.from_numpy(context_features),
            "num_voxels": coords.shape[0],
        }

def scintillator_collate_fn(batch):
    """Custom collate that stacks sparse coordinates with a batch index."""
    coords_list, feats_list = [], []
    labels, global_features = [], []
    n_vox = []

    for i, sample in enumerate(batch):
        c = sample["coords"]
        b_idx = torch.full((c.shape[0], 1), i, dtype=torch.int32)
        coords_list.append(torch.cat([b_idx, c], dim=1))
        feats_list.append(sample["feats"])
        labels.append(sample["label"])
        global_features.append(sample["global_features"])
        n_vox.append(sample["num_voxels"])

    return {
        "coords": torch.cat(coords_list, dim=0),
        "feats": torch.cat(feats_list, dim=0),
        "label": torch.stack(labels),
        "global_features": torch.stack(global_features),
        "num_voxels": n_vox,
        "batch_size": len(batch),
    }
