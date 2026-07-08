from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def runtime_report() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    device_index = torch.cuda.current_device() if cuda else None
    device = torch.cuda.get_device_properties(device_index) if cuda else None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda,
        "cuda_device": device.name if device else None,
        "cuda_device_index": device_index,
        "cuda_capability": (
            list(torch.cuda.get_device_capability(device_index)) if cuda else None
        ),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()) if cuda else False,
        "flash_attn": package_version("flash-attn"),
        "numpy": package_version("numpy"),
        "pyyaml": package_version("PyYAML"),
    }


def require_flash_runtime(precision: str) -> dict[str, Any]:
    report = runtime_report()
    errors = []
    if report["flash_attn"] is None:
        errors.append("flash-attn is not installed")
    if not report["cuda_available"]:
        errors.append("CUDA is not available in this process")
    capability = report["cuda_capability"]
    if capability is not None and tuple(capability) < (8, 0):
        errors.append(
            f"GPU compute capability {capability[0]}.{capability[1]} is below the "
            "Ampere-class target for this FlashAttention pipeline"
        )
    if precision == "bf16" and report["cuda_available"] and not report["bf16_supported"]:
        errors.append("The selected GPU does not report bf16 support")
    if errors:
        joined = "; ".join(errors)
        raise RuntimeError(
            f"FlashAttention runtime check failed: {joined}. Activate the existing "
            "`platon-flashattn` Conda environment and run on a compatible CUDA GPU. "
            "No package installation or fallback is attempted."
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the 2D pipeline runtime")
    parser.add_argument("--require-flash", action="store_true")
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()
    report = runtime_report()
    print(json.dumps(report, indent=2))
    if args.require_flash:
        require_flash_runtime(args.precision)
        print("FlashAttention runtime: ready")


if __name__ == "__main__":
    main()
