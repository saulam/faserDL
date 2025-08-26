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
from torch.nn.utils.rnn import pad_sequence 
from timm.models.layers import trunc_normal_
from MinkowskiEngine import MinkowskiConvolution
from timm.models.vision_transformer import Block
from .utils import get_3d_sincos_pos_embed, GlobalFeatureEncoderSimple, CylindricalHeadNormalized


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
        global_pool=False,
        metadata=None,
        **kwargs
    ):
        super(MinkViT, self).__init__(**kwargs)
        
        self.metadata = metadata
        norm_layer = kwargs['norm_layer']
        in_chans = kwargs['in_chans']
        drop_rate = kwargs['drop_rate']
        embed_dim = kwargs['embed_dim']
        patch_size = kwargs['patch_size']

        # patch & grid setup
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
        assert module_depth_voxels % p_d == 0, "module_depth_voxels must be divisible by patch depth"
        self.module_depth_voxels = module_depth_voxels
        self.module_depth_patches = module_depth_voxels // p_d
        G_h, G_w, G_d = self.grid_size
        assert G_d % self.module_depth_patches == 0, "grid depth must be multiple of module depth (in patches)"
        self.num_modules = G_d // self.module_depth_patches
        self.intra_grid_size = (G_h, G_w, self.module_depth_patches)     # H×W×(depth within module)
        self.num_intra_positions = G_h * G_w * self.module_depth_patches

        # patch embedding
        del self.patch_embed, self.pos_embed, self.norm_pre, self.fc_norm, self.head
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
        d_mod    = (DD % self.module_depth_patches).reshape(-1)           # [Np]
        module   = (DD // self.module_depth_patches).reshape(-1)          # [Np]
        intra    = (HH * (G_w * self.module_depth_patches)
                   + WW * self.module_depth_patches + d_mod).reshape(-1)  # [Np]

        self.register_buffer('idx_template',       flat_ids.long())       # [Np]
        self.register_buffer('intra_idx_template', intra.long())          # [Np]
        self.register_buffer('module_id_template', module.long())         # [Np]

        # experimenting
        self.global_feats_encoder = GlobalFeatureEncoderSimple(embed_dim)
        self.cross_attn   = nn.MultiheadAttention(embed_dim, kwargs['num_heads'], batch_first=True)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.intra_pos_embed = nn.Embedding(self.num_intra_positions, embed_dim) # frozen sin-cos
        self.module_embed_enc = nn.Embedding(self.num_modules, embed_dim)        # learned

        self.global_pool = global_pool
        if self.global_pool:
            embed_dim = kwargs['embed_dim']
            del self.norm  # remove the original norm

        self.head_channels = {
            "iscc": 1,
            "flavour": 3,
            "charm": 1,
            "vis": 3,
            "lep": 3,
        }
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

        self.apply(self._init_weights)

        # init heads
        for name in self.head_channels.keys():
            lin = self.heads[name][2]
            if name in self.metadata:
                # if head is a CylindricalHeadNormalized
                lin = lin.mlp
                trunc_normal_(lin.weight, std=2e-5)
                if lin.bias is not None:
                    with torch.no_grad():
                        lin.bias.copy_(torch.tensor([0., 0., 0., 0.], dtype=torch.float))
            else:
                trunc_normal_(lin.weight, std=2e-5)
                if lin.bias is not None:
                    nn.init.constant_(lin.bias, 0)


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

        idx       = self.idx_template.unsqueeze(0).expand(B, -1)
        intra_idx = self.intra_idx_template.unsqueeze(0).expand(B, -1)
        module_id = self.module_id_template.unsqueeze(0).expand(B, -1)

        return dense, idx, intra_idx, module_id
        

    def forward_features(self, x_sparse, glob):
        # patchify
        x_sparse = self.patch_embed(x_sparse)
        x, _, intra_idx, module_id = self.densify_patches(x_sparse)

        # add positional embeddings
        x = x + self.intra_pos_embed(intra_idx) + self.module_embed_enc(module_id)

        # add cls token
        #glob_emb = self.global_feats_encoder(glob).unsqueeze(1)  # (B, 1, D)
        #cls = self.cls_token + glob_emb   
        cls = self.cls_token.expand(x.size(0), -1, -1)                         
        x = torch.cat((cls, x), dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            outcome = x[:, 1:, :].mean(dim=1)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        # experimenting
        glob_emb = self.global_feats_encoder(glob).unsqueeze(1)  # (B, 1, D)
        y, _ = self.cross_attn(query=glob_emb, key=x, value=x)
        gate = torch.sigmoid(self.alpha)
        outcome = outcome + gate * y.squeeze(1)
        # end experimenting

        return outcome

    
    def forward_head(self, x):        
        outputs = {}
        for i, (name, head) in enumerate(self.heads.items()):
            output = head(x)
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
        embed_dim=384, patch_size=(48, 48, 2),
        depth=12, num_heads=12,
        mlp_ratio=4.0, qkv_bias=True, global_pool=True,
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
