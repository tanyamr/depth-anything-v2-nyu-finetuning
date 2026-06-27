"""Check local environment for Depth Anything V2 fine-tuning.

Example:
    python check_environment.py --config configs/strategies/train_vitb_head_only_mps.yaml
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")


def status_line(ok, label, detail):
    """Format one environment check line."""
    mark = "OK" if ok else "MISSING"
    return f"[{mark}] {label}: {detail}"


def check_path(path, label):
    """Print whether a path exists."""
    path = Path(path)
    print(status_line(path.exists(), label, path))
    return path.exists()


def check_python_packages():
    """Check core Python packages used by the project."""
    packages = [
        "torch",
        "torchvision",
        "yaml",
        "numpy",
        "PIL",
        "matplotlib",
        "tensorboard",
        "cv2",
        "tqdm",
    ]

    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "installed")
            print(status_line(True, package, version))
        except ImportError:
            print(status_line(False, package, "not installed"))


def check_torch_device():
    """Check CUDA/MPS availability."""
    try:
        import torch
    except ImportError:
        print(status_line(False, "torch device", "torch is not installed"))
        return

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    built_with_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
    print(status_line(torch.cuda.is_available(), "CUDA", torch.cuda.is_available()))
    print(status_line(built_with_mps, "MPS built", built_with_mps))
    print(status_line(has_mps, "MPS available", has_mps))


def load_config(config_path):
    """Load YAML config."""
    try:
        import yaml
    except ImportError:
        print(status_line(False, "pyyaml", "install dependencies first"))
        return None

    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/strategies/train_vitb_head_only_mps.yaml",
        help="Training config to validate",
    )
    args = parser.parse_args()

    print("Python packages")
    check_python_packages()

    print("\nDevices")
    check_torch_device()

    print("\nDepth Anything V2")
    check_path(
        "external/Depth-Anything-V2/metric_depth/depth_anything_v2/dpt.py",
        "official metric-depth code",
    )

    print("\nProject data and checkpoints")
    config = load_config(args.config)
    if config is None:
        return

    check_path(config["checkpoint_path"], "pretrained checkpoint")
    check_path(config["train_list"], "train split")
    check_path(config["val_list"], "val split")
    check_path(config["test_list"], "test split")

    print("\nConfig")
    print(status_line(True, "config", args.config))
    print(status_line(True, "device", config.get("device", "auto")))
    print(status_line(True, "strategy", config.get("finetune_strategy", "full")))


if __name__ == "__main__":
    main()
