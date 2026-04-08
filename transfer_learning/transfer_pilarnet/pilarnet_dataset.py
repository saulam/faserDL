"""Dataset utilities for PILArNet PID transfer."""

import hashlib
import json
import os
import threading
from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..transfer_charge_preprocessing import LogChargePreprocessor


PILARNET_PARTICLE_META_DIM = 31
PILARNET_BASE_PARTICLE_META_DIM = 25
PILARNET_SINGLE_PARTICLE_META_DIM = 16
PILARNET_SINGLE_META_FEATURE_IDX = np.asarray(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24],
    dtype=np.int64,
)

META_CENTROID = slice(0, 3)
META_BBOX = slice(3, 6)
META_START_POS = slice(15, 18)
META_PRINCIPAL_DIR = slice(18, 21)


@dataclass(frozen=True)
class XYOrientationAugmentation:
    rotate_k: int = 0
    flip_x: bool = False
    flip_y: bool = False


def _center_and_fit_with_scale(coords, spatial_shape):
    """Center on the bounding-box midpoint and downscale only when needed."""
    coords = coords.astype(np.float32, copy=False)
    spatial_shape = np.array(spatial_shape, dtype=np.float32)
    half_extent = (spatial_shape - 1.0) / 2.0

    mins = coords.min(axis=0, keepdims=True)
    maxs = coords.max(axis=0, keepdims=True)
    center = 0.5 * (mins + maxs)
    rel = coords - center

    max_abs = np.abs(rel).max(axis=0)
    safe_max_abs = np.maximum(max_abs, 1e-6)
    scale = min(1.0, float(np.min(half_extent / safe_max_abs)))

    centered = rel * scale + half_extent
    centered = np.rint(centered).astype(np.int32)
    spatial_max = np.array(spatial_shape, dtype=np.int32) - 1
    return np.clip(centered, 0, spatial_max), np.float32(scale)


def _compute_cluster_features(raw_coords, energy, grid_n, fit_scale):
    """Observable per-particle geometry features for event-level context."""
    coords = raw_coords.astype(np.float32, copy=False)
    energy = energy.astype(np.float32, copy=False)

    centroid = coords.mean(axis=0) / max(grid_n - 1, 1)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    bbox = (maxs - mins + 1.0) / float(grid_n)

    num_vox = float(len(coords))
    sum_energy = float(energy.sum())
    mean_energy = sum_energy / max(num_vox, 1.0)
    std_energy = float(energy.std()) if len(energy) > 1 else 0.0
    bbox_vol = float(np.prod(np.maximum(maxs - mins + 1.0, 1.0)))
    density = num_vox / max(bbox_vol, 1.0)

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
        proj = coords @ axis
        lo_idx = int(np.argmin(proj))
        hi_idx = int(np.argmax(proj))
        start_raw = coords[lo_idx].copy()
        end_raw = coords[hi_idx].copy()

        # Use a detector-axis ordering to fix the otherwise arbitrary PCA sign.
        if tuple(start_raw[[2, 0, 1]].tolist()) > tuple(end_raw[[2, 0, 1]].tolist()):
            start_raw, end_raw = end_raw, start_raw

        delta = end_raw - start_raw
        length = float(np.linalg.norm(delta)) / float(max(grid_n - 1, 1))
        if length > 1e-6:
            principal_dir = (delta / np.linalg.norm(delta)).astype(np.float32)
        start_pos = (start_raw / max(grid_n - 1, 1)).astype(np.float32)

        span = float(max(proj.max() - proj.min(), 1e-6))
        sqrt_evals = np.sqrt(np.maximum(evals, 0.0))
        linearity = float(
            np.log1p(sqrt_evals[0] / max(float(sqrt_evals[1]), 1e-6))
        )
        planarity = float(
            np.log1p(sqrt_evals[1] / max(float(sqrt_evals[2]), 1e-6))
        )
        midpoint = 0.5 * (proj.max() + proj.min())
        low_energy = float(energy[proj <= midpoint].sum())
        high_energy = float(energy[proj > midpoint].sum())
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
            float(fit_scale),
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


def _fractional_desc_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    n = values.size
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)
    order = np.argsort(-values, kind="stable")
    ranks = np.empty((n,), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=n, endpoint=True, dtype=np.float32)
    return ranks


def _append_event_context_features(meta_matrix, centroids, num_voxels, sum_energy):
    meta_matrix = np.asarray(meta_matrix, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)
    num_voxels = np.asarray(num_voxels, dtype=np.float32)
    sum_energy = np.asarray(sum_energy, dtype=np.float32)

    total_vox = float(np.maximum(num_voxels.sum(), 1.0))
    total_energy = float(np.maximum(sum_energy.sum(), 1e-6))
    vox_frac = num_voxels / total_vox
    charge_frac = sum_energy / total_energy
    vox_rank = _fractional_desc_ranks(num_voxels)
    charge_rank = _fractional_desc_ranks(sum_energy)

    event_centroid = centroids.mean(axis=0, keepdims=True)
    dist_event_centroid = np.linalg.norm(centroids - event_centroid, axis=1)
    dist_event_centroid /= np.sqrt(3.0)

    if len(centroids) > 1:
        pairwise = np.linalg.norm(
            centroids[:, None, :] - centroids[None, :, :],
            axis=-1,
        )
        np.fill_diagonal(pairwise, np.inf)
        nearest_dist = pairwise.min(axis=1)
    else:
        nearest_dist = np.ones((1,), dtype=np.float32)
    nearest_dist = nearest_dist / np.sqrt(3.0)

    event_features = np.stack(
        [
            charge_frac.astype(np.float32),
            vox_frac.astype(np.float32),
            charge_rank.astype(np.float32),
            vox_rank.astype(np.float32),
            nearest_dist.astype(np.float32),
            dist_event_centroid.astype(np.float32),
        ],
        axis=1,
    )
    return np.concatenate([meta_matrix, event_features], axis=1).astype(np.float32)


def _apply_xy_transform_to_unit_position(values, aug: XYOrientationAugmentation):
    values = np.asarray(values, dtype=np.float32).copy()
    x, y = float(values[0]), float(values[1])
    k = int(aug.rotate_k) % 4
    if k == 1:
        x, y = y, 1.0 - x
    elif k == 2:
        x, y = 1.0 - x, 1.0 - y
    elif k == 3:
        x, y = 1.0 - y, x
    if aug.flip_x:
        x = 1.0 - x
    if aug.flip_y:
        y = 1.0 - y
    values[0] = np.float32(np.clip(x, 0.0, 1.0))
    values[1] = np.float32(np.clip(y, 0.0, 1.0))
    return values


def _apply_xy_transform_to_direction(values, aug: XYOrientationAugmentation):
    values = np.asarray(values, dtype=np.float32).copy()
    x, y = float(values[0]), float(values[1])
    k = int(aug.rotate_k) % 4
    if k == 1:
        x, y = y, -x
    elif k == 2:
        x, y = -x, -y
    elif k == 3:
        x, y = -y, x
    if aug.flip_x:
        x = -x
    if aug.flip_y:
        y = -y
    values[0] = np.float32(x)
    values[1] = np.float32(y)
    return values


def _augment_particle_meta(meta, aug: XYOrientationAugmentation):
    meta = np.asarray(meta, dtype=np.float32).copy()
    if meta.shape[0] < PILARNET_BASE_PARTICLE_META_DIM:
        raise RuntimeError(
            f"Expected at least {PILARNET_BASE_PARTICLE_META_DIM} base particle features, got {meta.shape[0]}"
        )
    meta[META_CENTROID] = _apply_xy_transform_to_unit_position(meta[META_CENTROID], aug)
    meta[META_START_POS] = _apply_xy_transform_to_unit_position(meta[META_START_POS], aug)
    meta[META_PRINCIPAL_DIR] = _apply_xy_transform_to_direction(meta[META_PRINCIPAL_DIR], aug)
    if aug.rotate_k % 2 == 1:
        meta[[3, 4]] = meta[[4, 3]]
    return meta


def _select_single_particle_meta(meta):
    meta = np.asarray(meta, dtype=np.float32)
    if meta.shape[0] < PILARNET_BASE_PARTICLE_META_DIM:
        raise RuntimeError(
            f"Expected at least {PILARNET_BASE_PARTICLE_META_DIM} base particle features, got {meta.shape[0]}"
        )
    selected = meta[PILARNET_SINGLE_META_FEATURE_IDX]
    if selected.shape[0] != PILARNET_SINGLE_PARTICLE_META_DIM:
        raise RuntimeError(
            f"Expected single-particle metadata dim {PILARNET_SINGLE_PARTICLE_META_DIM}, got {selected.shape[0]}"
        )
    return selected.astype(np.float32, copy=False)


class PILArNetParticleDataset(Dataset):
    """Particle-level dataset backed by a prebuilt manifest and lazy HDF5 reads."""

    def __init__(
        self,
        manifest_path: str,
        spatial_shape: tuple[int, int, int] = (144, 144, 160),
        min_voxels: int = 5,
        augment: bool = False,
        cache_dir: str | None = None,
        grid_n: int = 768,
        charge_metadata_path: str | None = None,
        include_particle_meta: bool = False,
    ):
        data = np.load(manifest_path, allow_pickle=True)
        self.manifest = data["manifest"]
        self.h5_paths = list(data["h5_paths"])
        self.spatial_shape = spatial_shape
        self.min_voxels = min_voxels
        self.augment = augment
        self.cache_dir = cache_dir
        self.grid_n = grid_n
        self.cache_version = "bboxfit_v3"
        self.cache_namespace = self._build_cache_namespace(manifest_path)
        self.charge_preprocessor = LogChargePreprocessor(charge_metadata_path)
        self.include_particle_meta = include_particle_meta

        # Keep one HDF5 handle per worker thread.
        self._local = threading.local()

    def __len__(self):
        return len(self.manifest)

    def _get_h5(self, h5_idx: int) -> h5py.File:
        """Return an open HDF5 file handle, cached per worker thread."""
        cache = getattr(self._local, "h5_cache", None)
        if cache is None:
            cache = {}
            self._local.h5_cache = cache
        if h5_idx not in cache:
            cache[h5_idx] = h5py.File(self.h5_paths[h5_idx], "r")
        return cache[h5_idx]

    def _extract_particle_voxels(self, h5_idx, event_idx, particle_idx):
        """Load a single particle's voxels and energy values from HDF5."""
        f = self._get_h5(h5_idx)

        v_ext = f["Data/sparse3d_data_group/voxel_extents"][event_idx]
        vf, vn = int(v_ext["first"]), int(v_ext["N"])
        voxels = f["Data/sparse3d_data_group/voxels"][vf : vf + vn]

        g_ext = f["Data/sparse3d_group_group/voxel_extents"][event_idx]
        gf, gn = int(g_ext["first"]), int(g_ext["N"])
        groups = f["Data/sparse3d_group_group/voxels"][gf : gf + gn]["value"].astype(
            np.int32
        )

        mask = groups == particle_idx
        part_vox = voxels[mask]
        return part_vox

    def _decode_voxel_ids(self, voxel_ids):
        """Decode flat voxel IDs to (x, y, z) integer coordinates."""
        n = self.grid_n
        z = voxel_ids % n
        y = (voxel_ids // n) % n
        x = voxel_ids // (n * n)
        return np.stack([x, y, z], axis=1).astype(np.int64)

    def _center_and_fit(self, coords):
        """Center a particle cluster on its bounding-box midpoint."""
        centered, _ = _center_and_fit_with_scale(coords, self.spatial_shape)
        return centered

    def _sample_xy_orientation_augmentation(self):
        rotate_k = 0 if self.spatial_shape[0] != self.spatial_shape[1] else int(np.random.randint(0, 4))
        return XYOrientationAugmentation(
            rotate_k=rotate_k,
            flip_x=bool(np.random.rand() < 0.5),
            flip_y=bool(np.random.rand() < 0.5),
        )

    @staticmethod
    def _flip_xy(coords, spatial_shape, flip_x=False, flip_y=False):
        """Flip in-plane axes while preserving the detector drift axis."""
        S = np.array(spatial_shape, dtype=np.int32) - 1
        if flip_x:
            coords[:, 0] = S[0] - coords[:, 0]
        if flip_y:
            coords[:, 1] = S[1] - coords[:, 1]
        return coords

    @staticmethod
    def _rotate_xy_90(coords, spatial_shape, k):
        """Rotate in the x-y plane while preserving the detector drift axis."""
        if spatial_shape[0] != spatial_shape[1]:
            return coords
        k = int(k) % 4
        if k == 0:
            return coords

        Sx = int(spatial_shape[0] - 1)
        Sy = int(spatial_shape[1] - 1)
        out = coords.copy()
        x = coords[:, 0].copy()
        y = coords[:, 1].copy()
        if k == 1:
            out[:, 0] = y
            out[:, 1] = Sx - x
        elif k == 2:
            out[:, 0] = Sx - x
            out[:, 1] = Sy - y
        else:
            out[:, 0] = Sy - y
            out[:, 1] = x
        return out

    @staticmethod
    def _random_translate(coords, spatial_shape, max_shift=(6, 6, 5)):
        """Apply a small in-crop translation without clipping voxels at the borders."""
        coords = np.asarray(coords, dtype=np.int32)
        if coords.size == 0:
            return coords

        max_shift = np.asarray(max_shift, dtype=np.int32)
        spatial_max = np.asarray(spatial_shape, dtype=np.int32) - 1
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        low = -np.minimum(max_shift, mins)
        high = np.minimum(max_shift, spatial_max - maxs)

        shift = np.zeros((3,), dtype=np.int32)
        for d in range(3):
            lo = int(low[d])
            hi = int(high[d])
            if lo == hi:
                shift[d] = lo
            else:
                shift[d] = int(np.random.randint(lo, hi + 1))
        return coords + shift

    def _augment_coords(self, coords, orientation_aug=None):
        if orientation_aug is None:
            orientation_aug = self._sample_xy_orientation_augmentation()
        coords = self._rotate_xy_90(coords, self.spatial_shape, orientation_aug.rotate_k)
        coords = self._flip_xy(
            coords,
            self.spatial_shape,
            flip_x=orientation_aug.flip_x,
            flip_y=orientation_aug.flip_y,
        )
        coords = self._random_translate(coords, self.spatial_shape)
        return coords

    def _cache_path(self, idx):
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, self.cache_namespace, f"{idx}.npz")

    def _build_cache_namespace(self, manifest_path: str) -> str:
        """Tie cached crops to the manifest and crop geometry."""
        manifest_abs = os.path.abspath(manifest_path)
        try:
            st = os.stat(manifest_abs)
            manifest_meta = {
                "path": manifest_abs,
                "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
            }
        except OSError:
            manifest_meta = {"path": manifest_abs}

        payload = {
            "cache_version": self.cache_version,
            "manifest": manifest_meta,
            "spatial_shape": list(self.spatial_shape),
            "min_voxels": int(self.min_voxels),
            "grid_n": int(self.grid_n),
        }
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return f"{self.cache_version}_{digest}"

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        h5_idx = int(entry["h5_idx"])
        event_idx = int(entry["event_idx"])
        particle_idx = int(entry["particle_idx"])
        orientation_aug = self._sample_xy_orientation_augmentation() if self.augment else None

        cp = self._cache_path(idx)
        raw_coords = energy = None
        particle_meta = None
        if cp is not None and os.path.exists(cp):
            cached = np.load(cp)
            coords = cached["coords"]
            feats = cached["feats"]
            if self.include_particle_meta and "meta" in cached:
                particle_meta = cached["meta"]
        else:
            voxels = self._extract_particle_voxels(h5_idx, event_idx, particle_idx)
            raw_coords = self._decode_voxel_ids(voxels["id"])
            energy = voxels["value"].astype(np.float32)

            coords, fit_scale = _center_and_fit_with_scale(raw_coords, self.spatial_shape)
            feats = energy.reshape(-1, 1)

            coords, feats = self._deduplicate(coords, feats)
            if self.include_particle_meta:
                particle_meta = _compute_cluster_features(raw_coords, energy, self.grid_n, fit_scale)

            if cp is not None:
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                payload = {"coords": coords, "feats": feats}
                if particle_meta is not None:
                    payload["meta"] = particle_meta
                np.savez_compressed(cp, **payload)

        if self.include_particle_meta and particle_meta is None:
            voxels = self._extract_particle_voxels(h5_idx, event_idx, particle_idx)
            raw_coords = self._decode_voxel_ids(voxels["id"])
            energy = voxels["value"].astype(np.float32)
            _, fit_scale = _center_and_fit_with_scale(raw_coords, self.spatial_shape)
            particle_meta = _compute_cluster_features(raw_coords, energy, self.grid_n, fit_scale)
            if cp is not None:
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                np.savez_compressed(cp, coords=coords, feats=feats, meta=particle_meta)

        feats = self.charge_preprocessor.transform(feats)

        if self.augment:
            coords = self._augment_coords(coords, orientation_aug=orientation_aug)

        type_label = int(entry["type_label"])
        output = {
            "coords": torch.from_numpy(coords).int(),
            "feats": torch.from_numpy(feats.astype(np.float32)),
            "type_label": torch.tensor(type_label, dtype=torch.long),
            "num_voxels": coords.shape[0],
        }
        if self.include_particle_meta:
            if self.augment:
                particle_meta = _augment_particle_meta(particle_meta, orientation_aug)
            output["particle_meta"] = torch.from_numpy(
                _select_single_particle_meta(particle_meta)
            )
        return output

    @staticmethod
    def _deduplicate(coords, feats):
        """Merge voxels at the same grid cell (sum energy)."""
        _, inv, counts = np.unique(
            coords, axis=0, return_inverse=True, return_counts=True
        )
        if counts.max() == 1:
            return coords, feats
        n_unique = len(counts)
        new_feats = np.zeros((n_unique, feats.shape[1]), dtype=np.float32)
        np.add.at(new_feats, inv, feats)
        new_coords = np.zeros((n_unique, 3), dtype=coords.dtype)
        new_coords[inv] = coords
        return new_coords, new_feats


class PILArNetMultiParticleDataset(PILArNetParticleDataset):
    """Event-level dataset that yields all qualifying particles in one event."""

    def __init__(
        self,
        manifest_path: str,
        spatial_shape: tuple[int, int, int] = (144, 144, 160),
        min_voxels: int = 5,
        augment: bool = False,
        cache_dir: str | None = None,
        grid_n: int = 768,
        min_particles_per_event: int = 2,
        max_particles_per_event: int | None = None,
        charge_metadata_path: str | None = None,
    ):
        super().__init__(
            manifest_path=manifest_path,
            spatial_shape=spatial_shape,
            min_voxels=min_voxels,
            augment=augment,
            cache_dir=cache_dir,
            grid_n=grid_n,
            charge_metadata_path=charge_metadata_path,
        )
        self.min_particles_per_event = min_particles_per_event
        self.max_particles_per_event = max_particles_per_event
        self.cache_version = "bboxfit_multi_v3"
        self.cache_namespace = self._build_cache_namespace(manifest_path)
        (
            self._event_particle_indices,
            self._event_offsets,
        ) = self._build_event_index()

    def _build_event_index(self):
        order = np.lexsort((self.manifest["event_idx"], self.manifest["h5_idx"]))
        sorted_manifest = self.manifest[order]

        h5_idx = sorted_manifest["h5_idx"]
        event_idx = sorted_manifest["event_idx"]
        changes = np.nonzero(
            (h5_idx[1:] != h5_idx[:-1]) | (event_idx[1:] != event_idx[:-1])
        )[0] + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [len(order)]))

        offsets = [0]
        flat_particle_indices = []
        for s, e in zip(starts, ends):
            if int(e - s) < self.min_particles_per_event:
                continue
            flat_particle_indices.extend(order[s:e].tolist())
            offsets.append(len(flat_particle_indices))

        if len(offsets) == 1:
            raise RuntimeError("No events satisfy the multi-particle selection.")

        return (
            np.asarray(flat_particle_indices, dtype=np.int64),
            np.asarray(offsets, dtype=np.int64),
        )

    def __len__(self):
        return len(self._event_offsets) - 1

    def _extract_event_voxels(self, h5_idx, event_idx):
        """Load all voxels and particle-group labels for one event."""
        f = self._get_h5(h5_idx)

        v_ext = f["Data/sparse3d_data_group/voxel_extents"][event_idx]
        vf, vn = int(v_ext["first"]), int(v_ext["N"])
        voxels = f["Data/sparse3d_data_group/voxels"][vf : vf + vn]

        g_ext = f["Data/sparse3d_group_group/voxel_extents"][event_idx]
        gf, gn = int(g_ext["first"]), int(g_ext["N"])
        groups = f["Data/sparse3d_group_group/voxels"][gf : gf + gn]["value"].astype(
            np.int32
        )
        return voxels, groups

    def _select_event_entries(self, idx):
        start = int(self._event_offsets[idx])
        end = int(self._event_offsets[idx + 1])
        manifest_indices = self._event_particle_indices[start:end]
        entries = self.manifest[manifest_indices]

        if self.max_particles_per_event is not None and len(entries) > self.max_particles_per_event:
            keep = np.argsort(entries["num_voxels"])[::-1][: self.max_particles_per_event]
            manifest_indices = manifest_indices[keep]
            entries = entries[keep]

        sort_by_particle = np.argsort(entries["particle_idx"])
        return manifest_indices[sort_by_particle], entries[sort_by_particle]

    def __getitem__(self, idx):
        manifest_indices, entries = self._select_event_entries(idx)
        h5_idx = int(entries[0]["h5_idx"])
        event_idx = int(entries[0]["event_idx"])
        voxels, groups = self._extract_event_voxels(h5_idx, event_idx)
        orientation_aug = self._sample_xy_orientation_augmentation() if self.augment else None

        coords_list = []
        feats_list = []
        meta_list = []
        type_labels = []
        particle_ids = []
        raw_centroids = []
        raw_num_voxels = []
        raw_sum_energy = []

        for manifest_idx, entry in zip(manifest_indices, entries):
            particle_idx = int(entry["particle_idx"])
            cp = self._cache_path(int(manifest_idx))
            coords = feats = meta = None

            if cp is not None and os.path.exists(cp):
                cached = np.load(cp)
                coords = cached["coords"]
                feats = cached["feats"]
                meta = cached["meta"] if "meta" in cached else None

            if coords is None or feats is None or meta is None:
                part_vox = voxels[groups == particle_idx]
                raw_coords = self._decode_voxel_ids(part_vox["id"])
                energy = part_vox["value"].astype(np.float32)

                coords, fit_scale = _center_and_fit_with_scale(raw_coords, self.spatial_shape)
                feats = energy.reshape(-1, 1)
                coords, feats = self._deduplicate(coords, feats)
                meta = _compute_cluster_features(raw_coords, energy, self.grid_n, fit_scale)

                if cp is not None:
                    os.makedirs(os.path.dirname(cp), exist_ok=True)
                    np.savez_compressed(cp, coords=coords, feats=feats, meta=meta)

            feats = self.charge_preprocessor.transform(feats)

            if self.augment:
                coords = self._augment_coords(coords, orientation_aug=orientation_aug)
                meta = _augment_particle_meta(meta, orientation_aug)

            coords_list.append(torch.from_numpy(coords).int())
            feats_list.append(torch.from_numpy(feats.astype(np.float32)))
            meta_list.append(meta.astype(np.float32))
            type_labels.append(int(entry["type_label"]))
            particle_ids.append(particle_idx)
            raw_centroids.append(np.asarray(meta[:3], dtype=np.float32))
            raw_num_voxels.append(float(entry["num_voxels"]))
            raw_sum_energy.append(float(np.expm1(float(meta[7]))))

        meta_matrix = _append_event_context_features(
            np.stack(meta_list, axis=0),
            np.stack(raw_centroids, axis=0),
            np.asarray(raw_num_voxels, dtype=np.float32),
            np.asarray(raw_sum_energy, dtype=np.float32),
        )
        if meta_matrix.shape[1] != PILARNET_PARTICLE_META_DIM:
            raise RuntimeError(
                f"Expected particle_meta dim {PILARNET_PARTICLE_META_DIM}, got {meta_matrix.shape[1]}"
            )

        n_particles = len(type_labels)
        return {
            "coords_list": coords_list,
            "feats_list": feats_list,
            "particle_meta": torch.from_numpy(meta_matrix),
            "type_label": torch.tensor(type_labels, dtype=torch.long),
            "h5_idx": torch.full((n_particles,), h5_idx, dtype=torch.int32),
            "event_idx": torch.full((n_particles,), event_idx, dtype=torch.int32),
            "particle_idx": torch.tensor(particle_ids, dtype=torch.int32),
        }

def pilarnet_collate_fn(batch):
    """Custom collate that stacks sparse coordinates with a batch index."""
    coords_list, feats_list = [], []
    type_labels = []
    n_vox = []
    metas = []
    has_particle_meta = "particle_meta" in batch[0]

    for i, sample in enumerate(batch):
        c = sample["coords"]
        b_idx = torch.full((c.shape[0], 1), i, dtype=torch.int32)
        coords_list.append(torch.cat([b_idx, c], dim=1))
        feats_list.append(sample["feats"])
        type_labels.append(sample["type_label"])
        n_vox.append(sample["num_voxels"])
        if has_particle_meta:
            metas.append(sample["particle_meta"])

    collated = {
        "coords": torch.cat(coords_list, dim=0),
        "feats": torch.cat(feats_list, dim=0),
        "type_label": torch.stack(type_labels),
        "num_voxels": n_vox,
        "batch_size": len(batch),
    }
    if has_particle_meta:
        collated["particle_meta"] = torch.stack(metas, dim=0)
    return collated


def pilarnet_multiparticle_collate_fn(batch):
    """Flatten all particles from an event batch into one sparse particle batch."""
    coords_list, feats_list = [], []
    labels, metas = [], []
    h5_idx, event_idx, particle_idx = [], [], []
    event_offsets = [0]

    particle_batch_idx = 0
    for sample in batch:
        n_particles = int(sample["type_label"].numel())
        event_offsets.append(event_offsets[-1] + n_particles)

        for local_idx in range(n_particles):
            coords = sample["coords_list"][local_idx]
            b_idx = torch.full((coords.shape[0], 1), particle_batch_idx, dtype=torch.int32)
            coords_list.append(torch.cat([b_idx, coords], dim=1))
            feats_list.append(sample["feats_list"][local_idx])
            labels.append(sample["type_label"][local_idx])
            metas.append(sample["particle_meta"][local_idx])
            h5_idx.append(sample["h5_idx"][local_idx])
            event_idx.append(sample["event_idx"][local_idx])
            particle_idx.append(sample["particle_idx"][local_idx])
            particle_batch_idx += 1

    return {
        "coords": torch.cat(coords_list, dim=0),
        "feats": torch.cat(feats_list, dim=0),
        "type_label": torch.stack(labels),
        "particle_meta": torch.stack(metas, dim=0),
        "event_offsets": torch.tensor(event_offsets, dtype=torch.long),
        "num_events": len(batch),
        "batch_size": particle_batch_idx,
        "h5_idx": torch.stack(h5_idx),
        "event_idx": torch.stack(event_idx),
        "particle_idx": torch.stack(particle_idx),
    }
