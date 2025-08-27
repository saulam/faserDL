"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 07.25

Description: PyTorch ViT model with MinkowskiEngine patching.
"""

import torch
import torch.nn as nn
import MinkowskiEngine as ME
import timm.models.vision_transformer as vit
from functools import partial
from timm.models.layers import trunc_normal_
from MinkowskiEngine import MinkowskiConvolution
from timm.models.vision_transformer import Block
from .utils import get_3d_sincos_pos_embed, GlobalFeatureEncoderSimple, CrossAttention, CylindricalHeadNormalized


class MinkViT(vit.VisionTransformer):
    """ 
    Vision Transformer with MinkowskiEngine patching 
    and support for global average pooling
    """
    def __init__(
        self,
        D=3,
        img_size=(48, 48, 200),
        inter_depth=2,
        module_depth_voxels=20,
        global_pool=False,
        metadata=None,
        **kwargs
    ):
        super(MinkViT, self).__init__(**kwargs)
        
        self.metadata = metadata
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
        self.register_buffer('patch_size', torch.tensor(patch_size))

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
        print("Module token indices shape:", self.module_token_indices.shape)

        # Encoder: hierarchical ViT
        self.inter_depth = inter_depth
        self.module_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))        # per-module CLS (shared weights)
        self.intra_pos_embed = nn.Embedding(self.num_intra_positions, embed_dim)  # fixed sin-cos per-module
        self.module_embed_enc = nn.Embedding(self.num_modules, embed_dim)         # learned module index

        # Inter-module transformer over M CLS tokens
        self.global_token_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.global_feats_encoder = GlobalFeatureEncoderSimple(embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.inter_depth)]
        self.inter_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(self.inter_depth)
        ])
        self.inter_norm = norm_layer(embed_dim)
        
        # experimenting
        self.cross_attn = CrossAttention(
            embed_dim, num_heads=num_heads, qkv_bias=True, attn_drop=attn_drop_rate, proj_drop=drop_rate
        )
        self.global_pool = global_pool
        if self.global_pool:
            del self.inter_norm  # remove the original inter norm

        self.head_channels = {
            "iscc": 1,
            "flavour": 3,
            "charm": 1,
            "vis": 3,
            "lep": 3,
        }
        self.num_tasks = len(self.head_channels)
        self.task_tokens = nn.Parameter(torch.zeros(1, self.num_tasks, embed_dim))
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

        # init task tokens
        nn.init.normal_(self.task_tokens, std=.02)

        self.apply(self._init_weights)

        def _init_cross_attn(m):
            if isinstance(m, CrossAttention):
                trunc_normal_(m.q.weight, std=0.02)
                trunc_normal_(m.k.weight, std=0.02)
                trunc_normal_(m.v.weight, std=0.02)
                nn.init.zeros_(m.q.bias)
                nn.init.zeros_(m.k.bias)
                nn.init.zeros_(m.v.bias)
                trunc_normal_(m.proj.weight, std=0.02)
                nn.init.zeros_(m.proj.bias)
                # tiny gate to start as near-identity residual
                with torch.no_grad():
                    m.gamma.fill_(1e-4)

        self.apply(_init_cross_attn)

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

        intra_idx = self.intra_idx_template.unsqueeze(0).expand(B, -1)
        module_id = self.module_id_template.unsqueeze(0).expand(B, -1)

        return dense, intra_idx, module_id
    

    def _group_tokens_by_module(self, x):
        """
        x: [B, Np, C] dense tokens
        returns:
            x_mod: [B, M, Lm, C] re-ordered by module then intra-order
        """
        B, Np, C = x.shape
        M, Lm = self.num_modules, self.num_intra_positions
        idx = self.module_token_indices.unsqueeze(0).expand(B, -1, -1)  # [B, M, Lm]
        x_flat = torch.gather(
            x, 1,
            idx.reshape(B, M*Lm).unsqueeze(-1).expand(-1, -1, C)
        )  # [B, M*Lm, C]
        return x_flat.view(B, M, Lm, C)
        

    def forward_features(self, x_sparse, x_glob):
        # patchify
        x_sparse = self.patch_embed(x_sparse)
        x, intra_idx, module_id = self.densify_patches(x_sparse)

        # add positional embeddings
        x = x + self.intra_pos_embed(intra_idx) + self.module_embed_enc(module_id)

        # group to modules
        x_mod = self._group_tokens_by_module(x)
        B, M, Lk, C = x_mod.shape

        # Intra-module transformer with per-module CLS
        cls = self.module_cls_token.expand(B*M, 1, C)
        x_intra = x_mod.reshape(B*M, Lk, C)
        x_intra = torch.cat([cls, x_intra], dim=1)
        x_intra = self.pos_drop(x_intra)
        for blk in self.blocks:
            x_intra = blk(x_intra)
        x_intra = self.norm(x_intra)

        # retrieve CLS tokens
        cls_mod = x_intra[:, 0, :].reshape(B, M, C)        # [B, M, C]

        # Inter-module transformer over CLS tokens
        g = self.global_feats_encoder(x_glob)                                  # [B, C]
        g_tok = g.unsqueeze(1) + self.global_token_embed                       # [B, 1, C]
        mod_pos = self.module_embed_enc.weight.unsqueeze(0).expand(B, -1, -1)
        x_inter = torch.cat([g_tok, cls_mod + mod_pos], dim=1)                 # [B, 1+M, C]
        for blk in self.inter_blocks:
            x_inter = blk(x_inter)

        if self.global_pool:
            outcome = x_inter[:, 1:, :].mean(dim=1)
        else:
            x_inter = self.inter_norm(x_inter)
            task_q  = self.task_tokens.expand(B, -1, -1)
            outcome = task_q + self.cross_attn(task_q, x_inter)

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
        depth=10, inter_depth=2, num_heads=12,
        mlp_ratio=4.0, qkv_bias=True, global_pool=False,
        block_fn=Block,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(**kwargs):
    model = MinkViT(
        in_chans=1, D=3, img_size=(48, 48, 200),
        embed_dim=768, patch_size=(48, 48, 2),
        depth=12, num_heads=12,
        mlp_ratio=4.0, qkv_bias=True, global_pool=False,
        block_fn=Block,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    

def vit_large(**kwargs):
    model = MinkViT(
        in_chans=1, D=3, img_size=(48, 48, 200),
        embed_dim=1008, patch_size=(48, 48, 2),
        depth=24, num_heads=16,
        mlp_ratio=4.0, qkv_bias=True, global_pool=True,
        block_fn=Block,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(**kwargs):
    model = MinkViT(
        in_chans=1, D=3, img_size=(48, 48, 200),
        embed_dim=1296, patch_size=(48, 48, 2),
        depth=32, num_heads=16,
        mlp_ratio=4.0, qkv_bias=True, global_pool=True,
        block_fn=Block,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
