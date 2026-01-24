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


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=(), layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_intra_layers = len(getattr(model, "blocks", []))
    num_ahcal_layers = len(getattr(model, "ahcal_blocks", []))
    num_io_layers    = len(getattr(model, "lat_xattn_blocks", []))

    if hasattr(model, "latent_self_blocks"):
        assert num_io_layers == len(model.latent_self_blocks), \
            "lat_xattn_blocks and latent_self_blocks must have same length"

    parallel_depth = max(num_intra_layers, num_ahcal_layers, 1)
    num_layers = parallel_depth + 2 * num_io_layers + 2
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_intra_layers, num_ahcal_layers, num_io_layers, parallel_depth)
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


def _map_to_parallel_depth(i, L, parallel_depth):
    """
    Map a branch-local layer index i in [0..L-1] onto a shared depth axis [1..parallel_depth]
    by relative position. Ensures last layer of any branch maps to parallel_depth.
    """
    if L <= 1:
        return parallel_depth
    # relative position in [0..1] -> [1..parallel_depth]
    pos = (i + 1) / float(L)
    lid = int(round(pos * parallel_depth))
    return max(1, min(parallel_depth, lid))


def get_layer_id_for_vit(name, num_intra_layers, num_ahcal_layers, num_io_layers, parallel_depth):
    """
    Assign a parameter with its layer id (parallel branches share the same depth axis).
    """
    if name.startswith((
        "fcal_patch_embed", "ahcal_patch_embed",
        "module_cls_token", "ahcal_cls_token",
        "intra_pos_embed", "ahcal_pos_embed",
        "ecal_embed", "muon_state_embed", 
        "muon_spec_count_encoder", "muon_spec_embed",
        "kv_src_embed",
    )):
        return 0
    elif name.startswith("blocks."):
        i = int(name.split(".")[1])
        return _map_to_parallel_depth(i, num_intra_layers, parallel_depth)
    elif name.startswith("ahcal_blocks."):
        i = int(name.split(".")[1])
        return _map_to_parallel_depth(i, num_ahcal_layers, parallel_depth)
    elif name.startswith("norm") or name.startswith("ahcal_norm"):
        return parallel_depth
    elif name.startswith("muon_spec_xattn"):
        return parallel_depth
    elif name.startswith(("module_embed_enc", "tokens_norm")):
        return parallel_depth + 1
    elif name.startswith("lat_xattn_blocks."):
        i = int(name.split(".")[1])
        base = parallel_depth + 2
        return base + 2 * i
    elif name.startswith("latent_self_blocks."):
        i = int(name.split(".")[1])
        base = parallel_depth + 2
        return base + 2 * i + 1
    elif name.startswith(("task_tokens", "task_cross_attn", "latents_norm", "gamma")):
        return parallel_depth + 2 + 2 * num_io_layers
    else:
        return parallel_depth + 2 + 2 * num_io_layers
