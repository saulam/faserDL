"""Checkpoint loading for the scintillator transfer model."""

import torch


_KEY_REMAP = {
    "module_cls_token": "cls_token",
}

_KEY_PREFIX_REMAP = {
    "fcal_patch_embed.": "patch_embed.",
}

_SKIP_PREFIXES = (
    "ahcal_",
    "ecal_",
    "muon_",
    "kv_src_embed",
    "module_embed_enc",
    "intra_pos_embed",
    "heads.",
    "task_tokens",
    "decoder",
    "mask_token",
)


def load_pretrained_encoder(model, checkpoint_path, verbose=True):
    """Load compatible FASERCal weights into the adapted model."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("model.", "", 1): v for k, v in state.items()}

    target_sd = model.state_dict()

    loaded, skipped_shape, skipped_detector, missing, unexpected = [], [], [], [], []
    new_sd = {}

    def _remap_key(src_key):
        if src_key in _KEY_REMAP:
            return _KEY_REMAP[src_key]
        for old_prefix, new_prefix in _KEY_PREFIX_REMAP.items():
            if src_key.startswith(old_prefix):
                return f"{new_prefix}{src_key[len(old_prefix):]}"
        return src_key

    for src_key, src_val in state.items():
        if any(src_key.startswith(p) for p in _SKIP_PREFIXES):
            skipped_detector.append(src_key)
            continue

        dst_key = _remap_key(src_key)

        if dst_key not in target_sd:
            unexpected.append(src_key)
            continue

        if src_val.shape != target_sd[dst_key].shape:
            skipped_shape.append(
                f"{src_key} ({list(src_val.shape)}) -> {dst_key} ({list(target_sd[dst_key].shape)})"
            )
            continue

        new_sd[dst_key] = src_val
        loaded.append(f"{src_key} -> {dst_key}")

    for key in target_sd:
        if key not in new_sd:
            missing.append(key)

    model.load_state_dict(new_sd, strict=False)
    model._pretrained_loaded_keys = set(new_sd.keys())
    model._patch_embed_loaded = any(key.startswith("patch_embed") for key in new_sd)

    report = {
        "loaded": loaded,
        "skipped_shape": skipped_shape,
        "skipped_detector": skipped_detector,
        "missing": missing,
        "unexpected": unexpected,
    }

    if verbose:
        print("\n=== Checkpoint loading summary ===")
        print(f"  Loaded:            {len(loaded)}")
        print(f"  Skipped (shape):   {len(skipped_shape)}")
        print(f"  Skipped (detector):{len(skipped_detector)}")
        print(f"  Missing (reinit):  {len(missing)}")
        print(f"  Unexpected:        {len(unexpected)}")
        if skipped_shape:
            print("\n  Shape mismatches:")
            for item in skipped_shape:
                print(f"    {item}")
        if missing:
            print("\n  Reinitialised (not in checkpoint):")
            for item in missing:
                print(f"    {item}")
        if unexpected:
            print("\n  Unexpected keys in checkpoint (ignored):")
            for item in unexpected:
                print(f"    {item}")
        print()

    return report
