"""
Based on: https://github.com/facebookresearch/mae/blob/main/models_mae.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.vision_transformer import Block
from .mae_utils import PatchEmbed3D, get_3d_sincos_pos_embed, GlobalFeatureEncoder


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class MaskedAutoencoderViT3D(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone for 3D inputs (H, W, D)"""
    def __init__(
        self,
        img_size=(48, 48, 200),    # (H, W, D)
        patch_size=(16, 16, 10),   # (p_h, p_w, p_d)
        in_chans=1,
        embed_dim=384,
        depth=24,
        num_heads=16,
        decoder_embed_dim=192,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss=False,
        loss_weights={'occ': 1.0, 'charge': 1.0, 'lepton': 1.0, 'seg': 1.0},
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.loss_weights = loss_weights

        # patch embedding
        self.patch_embed = PatchEmbed3D(patch_size, in_chans, embed_dim)
        H, W, D = img_size
        p_h, p_w, p_d = patch_size
        assert H % p_h == 0 and W % p_w == 0 and D % p_d == 0, \
            "img_size must be divisible by patch_size"
        self.grid_size = (H // p_h, W // p_w, D // p_d)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.patch_voxels = p_h * p_w * p_d

        # MAE encoder specifics
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.global_feats_encoder = GlobalFeatureEncoder(embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.num_patches, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.num_patches, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred_occ = nn.Linear(decoder_embed_dim, self.patch_voxels, bias=True)
        self.decoder_pred_charge = nn.Linear(decoder_embed_dim, self.patch_voxels, bias=True)
        self.decoder_pred_lepton = nn.Linear(decoder_embed_dim, self.patch_voxels, bias=True)
        self.decoder_pred_seg = nn.Linear(decoder_embed_dim, self.patch_voxels * 3, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # init fixed pos embeddings
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos = get_3d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], self.grid_size, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos).float().unsqueeze(0))

        # init patch_embed (Conv3d) like a Linear (instead of nn.Conv3d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)

        # init tokens
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        # apply to other submodules
        self.apply(_init_weights)

    def patchify(self, imgs):
        """
        imgs: (B, C, H, W, D)
        returns x: (B, L, p_h*p_w*p_d*C)
        """
        B, C, H, W, D = imgs.shape
        p_h, p_w, p_d = self.patch_size
        g_h, g_w, g_d = self.grid_size
        # reshape to (B, C, g_h, p_h, g_w, p_w, g_d, p_d)
        x = imgs.reshape(
            B, C,
            g_h, p_h,
            g_w, p_w,
            g_d, p_d
        )
        # permute to (B, g_h, g_w, g_d, p_h, p_w, p_d, C)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)
        # flatten to patches
        x = x.reshape(B, g_h * g_w * g_d, p_h * p_w * p_d * C)
        return x

    def unpatchify(self, x):
        """
        x: (B, L, p_h*p_w*p_d*C)
        returns imgs: (B, C, H, W, D)
        """
        B, L, _ = x.shape
        p_h, p_w, p_d = self.patch_size
        g_h, g_w, g_d = self.grid_size
        C = _ // (p_h * p_w * p_d)
        x = x.reshape(B, g_h, g_w, g_d, p_h, p_w, p_d, C)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)  # (B, C, g_h, p_h, g_w, p_w, g_d, p_d)
        imgs = x.reshape(B, C, g_h * p_h, g_w * p_w, g_d * p_d)
        return imgs

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones(N, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, glob, mask_ratio):
        # x: (B, C, H, W, D)
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        glob_emb = self.global_feats_encoder(glob).unsqueeze(1)  # (B, 1, embed_dim)
        cls = self.cls_token + self.pos_embed[:, :1, :] + glob_emb
        x = torch.cat((cls, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        B, N, D = x.shape
        num_mask = ids_restore.shape[1] + 1 - N
        mask_tokens = self.mask_token.repeat(B, num_mask, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).repeat(1, 1, D))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        patch_features = x[:, 1:, :]
        preds = {
            'occ': self.decoder_pred_occ(patch_features),
            'charge': self.decoder_pred_charge(patch_features),
            'lepton': self.decoder_pred_lepton(patch_features),
            'seg': self.decoder_pred_seg(patch_features),
        }
        return preds

    def forward_loss(self, imgs, lepton_labels, seg_labels, preds, mask):
        """
        Calculates the multi-task loss.
        Loss is only computed for patches that were both masked and originally non-empty.
        """
        target_charge = self.patchify(imgs)
        target_lepton = self.patchify(lepton_labels)
        target_seg = self.patchify(seg_labels)
        target_occ = (target_charge > 0).float()

        if self.norm_pix_loss:
            mean = target_charge.mean(-1, keepdim=True)
            var = target_charge.var(-1, keepdim=True)
            target_charge = (target_charge - mean) / (var + 1.e-6)**.5

        min_vals, _ = target_charge.min(dim=-1)
        max_vals, _ = target_charge.max(dim=-1)
        nonempty_patch_mask = (min_vals != max_vals).float()  # (B, L)

        # The effective mask considers both the random masking and if the patch is non-empty
        effective_patch_mask = mask * nonempty_patch_mask
        # Expand to per-voxel mask for the occupancy loss
        effective_voxel_mask_occ = effective_patch_mask.unsqueeze(-1).expand_as(target_occ)
        # For other losses, we also require the specific voxel to be occupied (use target_occ for this)
        effective_voxel_mask_conditional = effective_voxel_mask_occ * target_occ

        total_loss = 0.
        losses = {}
        occ_mask_sum = effective_voxel_mask_occ.sum() + 1e-8
        conditional_mask_sum = effective_voxel_mask_conditional.sum() + 1e-8

        # Occupancy Loss (BCE)
        loss_occ = F.binary_cross_entropy_with_logits(preds['occ'], target_occ, reduction='none')
        loss_occ = (loss_occ * effective_voxel_mask_occ).sum() / occ_mask_sum
        losses['occ'] = loss_occ
        total_loss += loss_occ * self.loss_weights['occ']

        # Charge Loss (MSE)
        loss_charge = F.mse_loss(preds['charge'], target_charge, reduction='none')
        loss_charge = (loss_charge * effective_voxel_mask_conditional).sum() / conditional_mask_sum
        losses['charge'] = loss_charge
        total_loss += loss_charge * self.loss_weights['charge']

        # Lepton Loss (BCE)
        loss_lepton = F.binary_cross_entropy_with_logits(preds['lepton'], target_lepton, reduction='none')
        loss_lepton = (loss_lepton * effective_voxel_mask_conditional).sum() / conditional_mask_sum
        losses['lepton'] = loss_lepton
        total_loss += loss_lepton * self.loss_weights['lepton']

        # Particle Seg Loss (Cross-Entropy)
        B, L, P_x_C = preds['seg'].shape
        pred_seg_reshaped = preds['seg'].reshape(B * L * self.patch_voxels, 3)
        target_seg_reshaped = target_seg.reshape(B * L * self.patch_voxels, 3)
        loss_seg = F.cross_entropy(pred_seg_reshaped, target_seg_reshaped, reduction='none')
        loss_seg = loss_seg.reshape_as(target_occ) # Reshape to (B, L, P)
        loss_seg = (loss_seg * effective_voxel_mask_conditional).sum() / conditional_mask_sum
        losses['seg'] = loss_seg
        total_loss += loss_seg * self.loss_weights['seg']

        return total_loss, losses

    def forward(self, imgs, lepton_labels, seg_labels, glob, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, glob, mask_ratio)
        preds = self.forward_decoder(latent, ids_restore)
        total_loss, individual_losses = self.forward_loss(imgs, lepton_labels, seg_labels, preds, mask)
        return total_loss, preds, mask, individual_losses
