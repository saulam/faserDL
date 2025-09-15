"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 07.25

Description: PyTorch ViT model with MinkowskiEngine patching.
"""

import torch
import torch.nn as nn
#import MinkowskiEngine as ME
import timm.models.vision_transformer as vit
from functools import partial
from timm.models.layers import trunc_normal_
#from MinkowskiEngine import MinkowskiConvolution
from .utils import (
    get_3d_sincos_pos_embed, BlockWithMask, GlobalFeatureEncoderSimple, 
    CrossAttnBlock, CrossAttention, CylindricalHeadNormalized
)


class MinkViT(vit.VisionTransformer):
    """ 
    Vision Transformer with MinkowskiEngine patching 
    and support for global average pooling
    """
    def __init__(
        self,
        D=3,
        img_size=(48, 48, 200),
        module_depth_voxels=20,
        num_global_tokens=1,
        latent_tokens=32,
        io_depth=4,
        global_pool=False,
        metadata=None,
        **kwargs
    ):
        super(MinkViT, self).__init__(**kwargs)
        
        self.metadata = metadata
        depth = kwargs['depth']
        num_heads = kwargs['num_heads']
        mlp_ratio = kwargs['mlp_ratio']
        attn_drop_rate = kwargs['attn_drop_rate']
        drop_path_rate = kwargs['drop_path_rate']
        norm_layer = kwargs['norm_layer']
        in_chans = kwargs['in_chans']
        drop_rate = kwargs['drop_rate']
        embed_dim = kwargs['embed_dim']
        patch_size = kwargs['patch_size']

        # patch and grid setup
        H, W, D_img = img_size
        p_h, p_w, p_d = patch_size
        assert H % p_h == 0 and W % p_w == 0 and D_img % p_d == 0, \
            "img_size must be divisible by patch_size"
        self.grid_size = (H // p_h, W // p_w, D_img // p_d)
        self.num_patches = (
            self.grid_size[0] 
            * self.grid_size[1]
            * self.grid_size[2]
        )
        self.patch_voxels = p_h * p_w * p_d
        self.register_buffer('patch_size', torch.tensor(patch_size, dtype=torch.long))

        # module slicing along Z
        assert module_depth_voxels % p_d == 0, "module_depth_voxels must be divisible by patch depth"
        self.module_depth_voxels = module_depth_voxels
        self.module_depth_patches = module_depth_voxels // p_d
        G_h, G_w, G_d = self.grid_size
        assert G_d % self.module_depth_patches == 0, "grid depth must be multiple of module depth (in patches)"
        self.num_modules = G_d // self.module_depth_patches
        self.intra_grid_size = (G_h, G_w, self.module_depth_patches)     # H×W×(depth within module)
        self.num_intra_positions = G_h * G_w * self.module_depth_patches

        # patch embedding
        del self.cls_token, self.patch_embed, self.pos_embed, self.norm_pre, self.fc_norm, self.head
        self.patch_embed = MinkowskiConvolution(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, 
            bias=True, dimension=D,
        )

        # Precompute dense patch templates
        mh = torch.arange(G_h)
        mw = torch.arange(G_w)
        md = torch.arange(G_d)
        HH, WW, DD = torch.meshgrid(mh, mw, md, indexing='ij')

        flat_ids = (HH * (G_w * G_d) + WW * G_d + DD).reshape(-1)         # [Np]
        d_mod    = (DD % self.module_depth_patches)
        module   = (DD // self.module_depth_patches).reshape(-1)          # [Np]
        intra    = (HH * (G_w * self.module_depth_patches)
                   + WW * self.module_depth_patches + d_mod).reshape(-1)  # [Np]

        self.register_buffer('idx_template',       flat_ids.long())       # [Np]
        self.register_buffer('intra_idx_template', intra.long())          # [Np]
        self.register_buffer('module_id_template', module.long())         # [Np]

        # Per-module token index mapping [M, Lm]
        M = self.num_modules
        module_indices = []
        for m in range(M):
            mask = (self.module_id_template == m)
            intra_m = self.intra_idx_template[mask]
            flat_m  = self.idx_template[mask]
            order   = torch.argsort(intra_m)      # stable intra order
            module_indices.append(flat_m[order])
        self.register_buffer('module_token_indices', torch.stack(module_indices, 0))  # [M, Lm]

        # Encoder: hierarchical ViT
        total_depth = depth + 2 * io_depth - 1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        for idx, blk in enumerate(self.blocks):
            if hasattr(blk.drop_path1, "drop_prob") and hasattr(blk.drop_path2, "drop_prob"):
                blk.drop_path1.drop_prob = dpr[idx]
                blk.drop_path2.drop_prob = dpr[idx]
                
        self.module_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))        # per-module CLS (shared weights)
        self.intra_pos_embed = nn.Embedding(self.num_intra_positions, embed_dim)  # fixed sin-cos per-module
        self.module_embed_enc = nn.Embedding(self.num_modules, embed_dim)         # learned module index

        # Perceiver-IO bottleneck
        self.num_global_tokens = num_global_tokens
        self.global_feats_encoder = GlobalFeatureEncoderSimple(embed_dim, dropout=drop_rate)
        self.global_mem = nn.Parameter(torch.zeros(1, self.num_global_tokens, embed_dim))
        assert latent_tokens >= self.num_modules, \
            f"latent_tokens ({latent_tokens}) must be >= num_modules ({self.num_modules})"
        self.latent_tokens = latent_tokens
        self.latent_free_tokens = max(latent_tokens - self.num_modules, 0)
        self.latents_free = nn.Parameter(torch.zeros(1, self.latent_free_tokens, embed_dim))
        self.latent_xattn_blocks = nn.ModuleList([
            CrossAttnBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[depth + i*2 - 1] if i>0 else 0.,  # never drop the ingestion step
                norm_layer=norm_layer
            )
            for i in range(io_depth)
        ])
        self.latent_self_blocks = nn.ModuleList([
            BlockWithMask(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[depth + i*2], norm_layer=norm_layer
            )
            for i in range(io_depth)
        ])

        # Task specifics
        self.global_pool = global_pool
        self.head_channels = {
            "flavour": 4,
            "charm": 1,
            "vis": 3,
            "jet": 3,
        }
        self.num_tasks = len(self.head_channels)
        if not self.global_pool:
            # Task tokens and cross-attention
            self.latent_norm = norm_layer(embed_dim)
            self.task_tokens = nn.Parameter(torch.zeros(1, self.num_tasks, embed_dim))
            self.task_cross_attn = CrossAttention(
                dim=embed_dim, num_heads=num_heads, qkv_bias=True,
                attn_drop=attn_drop_rate,
            )
            self.gamma = nn.Parameter(torch.ones(1) * 1e-4) # scaling after cross-attention
        self.heads = nn.ModuleDict()
        for name in self.head_channels.keys():
            self.heads[name] = nn.Sequential(
                norm_layer(embed_dim),
                nn.Dropout(drop_rate),
                nn.Linear(embed_dim, self.head_channels[name]) 
                if name not in metadata 
                else self.make_head_from_stats(metadata[name], hidden=embed_dim)
            )
            
        self.initialize_weights()


    def initialize_weights(self):
        # init fixed pos embeddings
        enc_pos = get_3d_sincos_pos_embed(
            self.intra_pos_embed.weight.shape[-1],
            self.intra_grid_size,
            cls_token=False
        )
        with torch.no_grad():
            self.intra_pos_embed.weight.copy_(torch.from_numpy(enc_pos).float())
            self.intra_pos_embed.weight.requires_grad_(False)

        # init tokens
        with torch.no_grad():
            nn.init.normal_(self.global_mem, std=.02)
            nn.init.normal_(self.latents_free, std=.02)
            nn.init.normal_(self.module_cls_token, std=.02)
            nn.init.normal_(self.module_embed_enc.weight, std=0.02)
            if not self.global_pool:
                nn.init.normal_(self.task_tokens, std=.02)

        self.apply(self._init_weights)

        # init heads
        for name, head in self.heads.items():
            lin = head[2]
            if name in self.metadata and hasattr(lin, "mlp"):
                lin = lin.mlp
            trunc_normal_(lin.weight, std=2e-5)
            if lin.bias is not None:
                lin.bias.data.fill_(0)


    def _init_weights(self, m):
        if isinstance(m, MinkowskiConvolution):
            # initialize conv like nn.Linear
            w = m.kernel.data
            torch.nn.init.xavier_uniform_(w.view(-1, w.size(2)))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def no_weight_decay(self):
        return {
            'module_cls_token',
            'module_embed_enc.weight',
            'global_mem',
            'latent_free',
            'task_tokens',
            'patch_embed.bias',
        }
            

    def make_head_from_stats(self, stats_entry, hidden=128):
        """
        stats_entry: a dict like stats['vis'] or stats['lep']
                    with keys: mu_logpT, sigma_logpT, mu_pz, sigma_pz, eps_T
        """
        return CylindricalHeadNormalized(
            k_T=stats_entry["k_T"],
            mu_uT=stats_entry["mu_uT"],
            sigma_uT=stats_entry["sigma_uT"],
            k_Z=stats_entry["k_Z"],
            mu_uZ=stats_entry["mu_uZ"],
            sigma_uZ=stats_entry["sigma_uZ"],
            hidden=hidden,
        )


    def densify_patches(self, x_sparse):
        coords = x_sparse.C.long()  # [N,4]
        feats  = x_sparse.F

        B      = int(coords[:, 0].max().item()) + 1
        G_h, G_w, G_d = self.grid_size
        Np     = self.num_patches
        C      = feats.size(1)

        # convert from voxel coords to patch-grid coords
        h = coords[:, 1] // self.patch_size[0]
        w = coords[:, 2] // self.patch_size[1]
        d = coords[:, 3] // self.patch_size[2]

        patch_ids = (h * (G_w * G_d)) + (w * G_d) + d
        batch_ids = coords[:, 0]
        
        # scatter into dense tensor
        dense = feats.new_zeros(B, Np, C)
        dense[batch_ids, patch_ids, :] = feats

        attn_mask = torch.zeros(B, Np, dtype=torch.bool, device=feats.device)
        attn_mask[batch_ids, patch_ids] = True
        intra_idx = self.intra_idx_template.unsqueeze(0).expand(B, -1)

        return dense, attn_mask, intra_idx
    

    def _group_tokens_by_module(self, x, attn_mask):
        """
        x: [B, Np, C] dense tokens
        attn_mask: [B, Np]  (bool)
        returns:
            x_mod: [B, M, Lm, C] re-ordered by module then intra-order
            attn_mask_mod: [B, M, Lm] (bool)
        """
        B, Np, C = x.shape
        M, Lm = self.num_modules, self.num_intra_positions
        idx = self.module_token_indices.unsqueeze(0).expand(B, -1, -1)  # [B, M, Lm]
        flat_idx = idx.reshape(B, M * Lm)

        x_mod = torch.gather(x, 1, flat_idx.unsqueeze(-1).expand(-1, -1, C)).contiguous()  # [B, M*Lm, C]
        attn_mask_mod = torch.gather(attn_mask, 1, flat_idx).contiguous()                  # [B, M*Lm]
        return x_mod.view(B, M, Lm, C), attn_mask_mod.view(B, M, Lm)


    def _pack_tokens(self, kv, attn_mask):
        """
        kv:             [B, M, L, C]
        attn_mask:      [B, M, L]
        returns:
            kv_tokens:  [B, N_max, C] (packed real tokens, padded per batch)
            kv_mask:    [B, N_max]
        """
        B, M, L, C = kv.shape
        device = kv.device

        b_ids, m_ids, lk_ids = torch.nonzero(attn_mask, as_tuple=True)  # [Nk]
        N = b_ids.numel()

        # per-batch ranks to map ragged -> padded
        _, counts = torch.unique_consecutive(b_ids, return_counts=True)
        N_max = int(counts.max().item())
        starts = torch.cumsum(counts, dim=0) - counts
        within = torch.arange(N, device=device) - torch.repeat_interleave(starts, counts)

        kv_tokens = kv.new_zeros(B, N_max, C)
        kv_mask   = torch.zeros(B, N_max, dtype=torch.bool, device=device)
        kv_tokens[b_ids, within] = kv[b_ids, m_ids, lk_ids, :]   # pack only real kept
        kv_mask[b_ids, within]   = True
        return kv_tokens, kv_mask
    

    def _prepare_kv_tokens(
        self,
        kv,
        attn_mask,
        pack: bool = True,
        adaptive: bool = True,
        abs_threshold: int = 64,
        ratio_threshold: float = 1.5
    ):
        """
        Either reshape (fast) or pack (slow, but memory efficient) the kv tokens for Perceiver cross-attn.

        kv:              [B, M, L, C]
        attn_mask:       [B, M, L]
        pack:            if False -> always reshape; if True -> pack (optionally adaptive)
        adaptive:        if True, only pack when it clearly shrinks KV (see thresholds)
        abs_threshold:   pack if N_fixed - N_real_max >= this
        ratio_threshold: pack if N_fixed / N_real_max >= this

        returns:
            kv_tokens: [B, N(_max), C]
            kv_mask:   [B, N(_max)]
        """
        B, M, L, C = kv.shape

        if not pack:
            return kv.reshape(B, M * L, C), attn_mask.reshape(B, M * L)

        if adaptive:
            N_fixed = M * L
            N_real_max = attn_mask.reshape(B, -1).sum(dim=-1).max()
            # avoid div-by-zero if a degenerate empty batch appears
            ratio = N_fixed / max(int(N_real_max.item()), 1)
            if (N_fixed - int(N_real_max.item()) < abs_threshold) and (ratio < ratio_threshold):
                # not worth packing -> take the simple fast path
                return kv.reshape(B, N_fixed, C), attn_mask.reshape(B, N_fixed)

        # pack only real tokens
        return self._pack_tokens(kv, attn_mask)
    

    def _prepare_queries_and_mask(
        self,
        cls_mod: torch.Tensor,      # [B, M, C]  per-module CLS
        kv_keep: torch.Tensor,      # [B, N]    bool, real K/V positions
        *,
        adaptive: bool = True,
        alpha: float = 1.0,         # free queries ~ alpha * sqrt(#KV)
        min_free: int = 2,          # minimum number of free queries when adaptive
        jitter: int = 1,            # ± jitter on active free queries when training
    ):
        """
        Returns:
        queries      [B, Q, C]   (Q = M + K_free)
        q_mask       [B, Q]      bool, which queries are active
        pair_mask    [B, Q, N]   bool, (active query) AND (real KV)
        """
        B, M, C = cls_mod.shape
        device  = cls_mod.device
        Q_total = self.latent_tokens

        # Anchored CLS queries
        mod_ids   = torch.arange(M, device=device)
        q_anchor  = cls_mod + self.module_embed_enc(mod_ids).view(1, M, C)     # [B, M, C]

        # Free queries (K_free)
        if self.latent_free_tokens > 0:
            q_free = self.latents_free.expand(B, self.latent_free_tokens, -1)  # [B, K_free, C]
            queries = torch.cat([q_anchor, q_free], dim=1)                     # [B, Q_total, C]
        else:
            queries = q_anchor                                                 # [B, M, C]

        # Query mask
        q_mask = torch.zeros(B, Q_total, dtype=torch.bool, device=device)
        q_mask[:, :M] = True  # anchors always active

        if self.latent_free_tokens > 0:
            if adaptive:
                # how many free queries to activate per event (0..K_free)
                v_b = kv_keep.sum(dim=1).float()                  # [B]
                k_b = (alpha * v_b.sqrt()).round().clamp_(min_free, self.latent_free_tokens).long()
                if self.training and jitter > 0:
                    k_b = (k_b + torch.randint(-jitter, jitter + 1, k_b.shape, device=device)).clamp_(min_free, self.latent_free_tokens)

                num = int(k_b.max().item())
                if num > 0:
                    # sample which free slots to activate (ensures tail slots get trained over time)
                    idx  = torch.rand(B, self.latent_free_tokens, device=device).topk(num, dim=1, largest=False, sorted=False).indices  # [B,num]
                    take = (torch.arange(num, device=device).unsqueeze(0) < k_b.unsqueeze(1))                           # [B,num]
                    r, c = take.nonzero(as_tuple=True)
                    q_mask[r, M + idx[r, c]] = True
            else:
                q_mask[:, :] = True

        # Single pair mask for CrossAttnBlock
        pair_mask = q_mask.unsqueeze(-1) & kv_keep.unsqueeze(1)  # [B, Q, N]

        return queries, q_mask, pair_mask
    

    def forward_features(self, x_sparse, x_glob):
        # patchify
        x_sparse = self.patch_embed(x_sparse)
        x, attn_mask, intra_idx = self.densify_patches(x_sparse)

        # add positional embeddings
        x = x + self.intra_pos_embed(intra_idx)

        # group to modules
        x_mod, attn_mask_mod = self._group_tokens_by_module(x, attn_mask)
        B, M, L, C = x_mod.shape

        # Intra-module transformer with per-module CLS
        cls = self.module_cls_token.expand(B*M, 1, C)
        x_intra = x_mod.reshape(B*M, L, C)
        x_intra = torch.cat([cls, x_intra], dim=1)
        x_intra = self.pos_drop(x_intra)
        attn_mask_intra = torch.cat([torch.ones(B*M, 1, dtype=torch.bool, device=x.device), attn_mask_mod.reshape(B*M, L)], dim=1)
        for blk in self.blocks:
            x_intra = blk(x_intra, attn_mask=attn_mask_intra)
        x_intra = self.norm(x_intra)

        # discard intra-module CLS and add module embedding
        cls_mod = x_intra[:, 0, :].view(B, M, C)                      # [B, M, C]
        tok_mod = x_intra[:, 1:, :].reshape(B, M, L, C)               # [B, M, L, C]
        mod_ids = torch.arange(self.num_modules, device=tok_mod.device)
        tok_mod = tok_mod + self.module_embed_enc(mod_ids).view(1, M, 1, -1)

        # global input
        g_enc   = self.global_feats_encoder(x_glob)                   # [B, C]
        g_tokens = g_enc.unsqueeze(1) + self.global_mem               # [B, G, C]
        g_tokens = g_tokens.to(tok_mod.dtype)

        # Perceiver-IO encoder: latents attend to all kept tokens
        kv_tokens, kv_keep = self._prepare_kv_tokens(tok_mod, attn_mask_mod)  # [B, N(_max), C], [B, N(_max)]

        # append global tokens to kv
        kv_tokens = torch.cat([kv_tokens, g_tokens], dim=1)                   # [B, N(_max)+G, C]
        kv_keep  = torch.cat([kv_keep, torch.ones(
            kv_keep.size(0), self.num_global_tokens, 
            dtype=torch.bool, device=kv_keep.device)], dim=1)                 # [B, N(_max)+G]
        
        # prepare queries and masks
        queries, q_mask, pair_mask = self._prepare_queries_and_mask(
            cls_mod, kv_keep, adaptive=False,
        )
        lat = queries
        for xa, sa in zip(self.latent_xattn_blocks, self.latent_self_blocks):
            lat = xa(lat, kv_tokens, attn_mask=pair_mask)                     # cross: lat <- tokens
            lat = sa(lat, attn_mask=q_mask)                                   # self-attn over latents

        # Task-specific heads
        if self.global_pool:
            outcome = lat.mean(dim=1)
        else:
            task_q  = self.task_tokens.expand(B, -1, -1)
            lat = self.latent_norm(lat)
            outcome = task_q + self.task_cross_attn(task_q, lat) * self.gamma

        return outcome
    

    def forward_head(self, x):
        outputs = {}
        for i, (name, head) in enumerate(self.heads.items()):
            if self.global_pool:
                output = head(x)   
            else:
                output = head(x[:, i, :])
            outputs[f"out_{name}"] = output

        return outputs
        
        
    def forward(self, x, x_glob):
        x = self.forward_features(x, x_glob)
        x = self.forward_head(x)
        return x

        
    def replace_depthwise_with_channelwise(self):
        """
        Replace all MinkowskiDepthwiseConvolution modules with
        MinkowskiChannelwiseConvolution modules preserving the parameters.
        """
        for name, module in self.named_modules():
            if isinstance(module, ME.MinkowskiDepthwiseConvolution):
                # Retrieve parameters of the current depthwise convolution
                in_channels = module.in_channels
                kernel_size = module.kernel_generator.kernel_size
                stride = module.kernel_generator.kernel_stride
                dilation = module.kernel_generator.kernel_dilation
                bias = module.bias is not None
                dimension = module.dimension
                
                # Create a new channelwise convolution with the same parameters
                new_conv = ME.MinkowskiChannelwiseConvolution(
                    in_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    bias=bias,
                    dimension=dimension
                )
                new_conv.kernel = module.kernel
                if bias:
                    new_conv.bias = module.bias
                
                # Replace the old module with the new one
                parent_module, attr_name = self._get_parent_module(name)
                setattr(parent_module, attr_name, new_conv)
        
        return

    
    def _get_parent_module(self, layer_name):
        """
        Retrieve the parent module and attribute name for a given layer.
        
        Args:
            layer_name (str): Dot-separated module path.
        
        Returns:
            Tuple of (parent_module, attribute_name)
        """
        components = layer_name.split('.')
        parent = self
        for comp in components[:-1]:
            parent = getattr(parent, comp)
        return parent, components[-1]


def vit_tiny(**kwargs):
    model = MinkViT(
        in_chans=1, D=3, img_size=(48, 48, 200),
        embed_dim=528, patch_size=(16, 16, 4),
        depth=2, num_heads=12, num_global_tokens=1,
        latent_tokens=16, io_depth=2,
        mlp_ratio=4.0, qkv_bias=True, global_pool=True,
        block_fn=BlockWithMask,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(**kwargs):
    model = MinkViT(
        in_chans=1, D=3, img_size=(48, 48, 200),
        embed_dim=768, patch_size=(16, 16, 4),
        depth=4, num_heads=12, num_global_tokens=1,
        latent_tokens=32, io_depth=4,
        mlp_ratio=4.0, qkv_bias=True, global_pool=True,
        block_fn=BlockWithMask,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    

def vit_large(**kwargs):
    model = MinkViT(
        in_chans=1, D=3, img_size=(48, 48, 200),
        embed_dim=1008, patch_size=(48, 48, 2),
        depth=8, num_heads=12, num_global_tokens=2,
        latent_tokens=32, io_depth=8,
        mlp_ratio=4.0, qkv_bias=True, global_pool=True,
        block_fn=BlockWithMask,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(**kwargs):
    model = MinkViT(
        in_chans=1, D=3, img_size=(48, 48, 200),
        embed_dim=1296, patch_size=(48, 48, 2),
        depth=16, num_heads=12, num_global_tokens=4,
        latent_tokens=64, io_depth=16,
        mlp_ratio=4.0, qkv_bias=True, global_pool=True,
        block_fn=BlockWithMask,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
