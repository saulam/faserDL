# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_intra_layers = len(model.blocks)
    num_self_layers  = len(model.latent_self_blocks)
    num_xattn_layers = 2 * num_self_layers
    num_layers = num_intra_layers + num_xattn_layers + num_self_layers + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_intra_layers, num_xattn_layers, num_self_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)
    
    print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_intra_layers, num_xattn_layers, num_self_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name.startswith(("patch_embed", "module_cls_token", "intra_pos_embed")):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    elif name.startswith('norm'):
        return num_intra_layers
    elif name.startswith(("module_embed_enc", "global_feats_encoder", "global_mem")):
        return num_intra_layers + 1
    elif name.startswith("xattn_blocks."):
        parts = name.split(".")
        which = parts[1]
        base = num_intra_layers + 1 + 3 * int(parts[2])
        if "lat<-tok" in which:
            return base + 0
        else:  # "tok<-lat"
            return base + 2
    elif name.startswith("latent_self_blocks."):
        return num_intra_layers + 1 + 3 * int(name.split(".")[1]) + 1
    elif name.startswith('tokens_norm'):
        return num_intra_layers + num_xattn_layers + num_self_layers
    else:
        return num_intra_layers + num_xattn_layers + num_self_layers + 1
