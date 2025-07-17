import torch
from typing import Sequence, List, Tuple

class SparseVolumePatcher:
    """
    A utility class for patchifying and unpatchifying sparse 3D volumes,
    now supporting multiple per-coordinate value tensors of arbitrary shape,
    with patches always returned as 2D tensors.

    Attributes:
        volume_shape (tuple[int, int, int]): The full volume dimensions (H, W, D).
        patch_size (tuple[int, int, int]): The patch dimensions along each axis.
        num_patches (tuple[int, int, int]): Number of patches along each axis.
        patch_volume (int): Number of voxels in a single patch.
    """

    def __init__(
        self,
        patch_size: int | tuple[int, int, int],
        volume_shape: tuple[int, int, int] = (48, 48, 200),
    ):
        # Normalize and validate patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3
        if len(patch_size) != 3:
            raise ValueError("patch_size must be an int or a tuple of three ints.")
        self.volume_shape = volume_shape
        self.patch_size = patch_size

        # Compute number of patches per axis
        self.num_patches = tuple(
            vs // ps for vs, ps in zip(self.volume_shape, self.patch_size)
        )
        if any(vs % ps for vs, ps in zip(self.volume_shape, self.patch_size)):
            raise ValueError(
                f"patch_size {self.patch_size} must exactly divide volume_shape {self.volume_shape}"
            )

        # Precompute strides
        nx, ny, nz = self.num_patches
        px, py, pz = self.patch_size
        self._patch_stride = (ny * nz, nz, 1)
        self._local_stride = (py * pz, pz, 1)
        self.patch_volume = px * py * pz

    def patchify(
        self,
        coords: torch.LongTensor,
        values: Sequence[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], torch.LongTensor]:
        """
        Extract non-empty patches for K different value-tensors per coordinate.

        Args:
            coords: (N,3) tensor of (x,y,z) indices in the volume.
            values: length-K list of (N, *S_k) tensors, one per channel.

        Returns:
            patches: length-K list of 2D tensors, each of shape (M, P_flat),
                     where M = # non-empty patches,
                     and P_flat = P * product(S_k).
            patch_ids: (M,) tensor of unique patch IDs (flat indices).
        """
        coords = coords.long()
        N = coords.size(0)
        K = len(values)
        if any(v.shape[0] != N for v in values):
            raise ValueError("All value-tensors must have the same first dimension as coords")

        device = coords.device
        psize_t = torch.tensor(self.patch_size, device=device)
        pstride_t = torch.tensor(self._patch_stride, device=device)
        lstride_t = torch.tensor(self._local_stride, device=device)

        # Compute flat patch IDs per coordinate
        patch_coords = coords // psize_t
        flat_patch_idx = (patch_coords * pstride_t).sum(dim=1)
        unique_ids, inv = torch.unique(flat_patch_idx, sorted=True, return_inverse=True)
        M = unique_ids.numel()

        # Local voxel index within each patch
        local = coords % psize_t
        local_idx = (local * lstride_t).sum(dim=1)

        patches: List[torch.Tensor] = []
        P = self.patch_volume
        # Build patches for each value tensor
        for v in values:
            # v: (N, *S_k)
            extra_shape = v.shape[1:]
            extra_size = int(v.numel() // N)
            P_flat = P * extra_size

            # Initialize patch matrix (M, P_flat)
            patch_k = torch.zeros((M, P_flat), dtype=v.dtype, device=device)
            # Flatten for scattering
            flat_patch = patch_k.view(-1)
            flat_vals = v.view(N, extra_size)

            # Compute base indices in flat_patch for voxel positions
            # Each patch row has P_flat entries, so patch offset = inv * P_flat
            # Within-patch offset = local_idx * extra_size
            base_idx = inv * P_flat + local_idx * extra_size

            if extra_size > 1:
                offsets = torch.arange(extra_size, device=device).unsqueeze(0)  # (1, extra_size)
                idx = base_idx.unsqueeze(1) + offsets  # (N, extra_size)
                idx1 = idx.view(-1)
                flat_patch[idx1] = flat_vals.view(-1)
            else:
                flat_patch[base_idx] = flat_vals.view(-1)

            patches.append(patch_k)

        return patches, unique_ids

    def unpatchify(
        self,
        patches: Sequence[torch.Tensor],
        patch_ids: torch.LongTensor,
    ) -> Tuple[torch.LongTensor, List[torch.Tensor]]:
        """
        Reconstruct a shared list of coords and per-channel values from 2D patches.

        Args:
            patches: length-K list of (M, P_flat) tensors, where P_flat = P * extra_size.
            patch_ids: (M,) tensor of flat patch IDs from `patchify`.

        Returns:
            coords: (L,3) tensor of recovered indices of all non-zero voxels across channels.
            values_list: length-K list of (L, extra_size) tensors with channel values,
                         or (L,) if extra_size=1 for that channel.
        """
        K = len(patches)
        M = patch_ids.numel()
        P = self.patch_volume

        device = patches[0].device
        psize_t = torch.tensor(self.patch_size, device=device)
        lstr = self._local_stride

        # Gather all non-zero flat positions across channels
        all_idx = torch.tensor([], dtype=torch.long, device=device)
        extra_sizes: List[int] = []
        for patch_k in patches:
            P_flat = patch_k.shape[1]
            extra_size = P_flat // P
            extra_sizes.append(extra_size)

            flat_k = patch_k.view(-1, extra_size) if extra_size>1 else patch_k.view(-1)
            nz = flat_k.nonzero(as_tuple=False)
            # if multi-col, nz is (idx, col)
            idx = nz[:,0] if extra_size>1 else nz.squeeze(1)
            all_idx = torch.cat((all_idx, idx))
        all_idx = all_idx.unique(sorted=True)
        L = all_idx.numel()

        # Compute coords for these L positions
        patch_idx = all_idx // (P * extra_sizes[0])  # assume same P for coords
        local_idx_flat = all_idx % (P * extra_sizes[0])
        # recover voxel index within patch
        local_voxel = local_idx_flat // extra_sizes[0] if extra_sizes[0]>1 else local_idx_flat

        base = patch_ids[patch_idx]
        pc = torch.stack([
            base // (self.num_patches[1] * self.num_patches[2]),
            (base // self.num_patches[2]) % self.num_patches[1],
            base % self.num_patches[2]
        ], dim=1)
        patch_origin = pc * psize_t

        lx = local_voxel // lstr[0]
        rem = local_voxel % lstr[0]
        ly = rem // lstr[1]
        lz = rem % lstr[1]
        local_coords = torch.stack((lx, ly, lz), dim=1)

        coords = patch_origin + local_coords

        # Extract per-channel values
        values_list: List[torch.Tensor] = []
        for k, patch_k in enumerate(patches):
            extra_size = extra_sizes[k]
            flat_k = patch_k.view(-1, extra_size) if extra_size>1 else patch_k.view(-1)
            vals_k = flat_k[all_idx]
            values_list.append(vals_k)

        return coords, values_list
