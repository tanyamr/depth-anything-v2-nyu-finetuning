"""Load Depth Anything V2 for fine-tuning.

- vits: checkpoints/pretrained/depth_anything_v2_metric_hypersim_vits.pth
- vitb: checkpoints/pretrained/depth_anything_v2_metric_hypersim_vitb.pth
- vitl: checkpoints/pretrained/depth_anything_v2_metric_hypersim_vitl.pth

"""

from pathlib import Path
import sys

import torch
from torch import nn


model_configs = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}


class LoRALinear(nn.Module):
    """Linear layer with a frozen base weight and trainable low-rank update."""

    def __init__(self, base_layer, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear can only wrap torch.nn.Linear layers.")

        self.base = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        self.dropout = nn.Dropout(float(dropout))

        self.lora_down = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, base_layer.out_features, bias=False)
        self.lora_down = self.lora_down.to(
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
        )
        self.lora_up = self.lora_up.to(
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
        )

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        base_output = self.base(x)
        lora_output = self.lora_up(self.lora_down(self.dropout(x))) * self.scaling
        return base_output + lora_output


def import_depth_anything_v2():
    """Import official Depth Anything V2 code.

    The official metric-depth code is usually cloned as:
        external/Depth-Anything-V2/metric_depth/

    This helper also checks a few common local locations before showing an
    installation hint.
    """
    try:
        from depth_anything_v2.dpt import DepthAnythingV2

        return DepthAnythingV2
    except ImportError:
        pass

    possible_paths = [
        Path("external/Depth-Anything-V2/metric_depth"),
        Path("Depth-Anything-V2/metric_depth"),
        Path("metric_depth"),
    ]

    for path in possible_paths:
        if (path / "depth_anything_v2" / "dpt.py").exists():
            sys.path.insert(0, str(path.resolve()))
            from depth_anything_v2.dpt import DepthAnythingV2

            return DepthAnythingV2

    raise ImportError(
        "Could not import DepthAnythingV2.\n"
        "Clone the official repository and install its metric-depth dependencies:\n"
        "  mkdir -p external\n"
        "  git clone https://github.com/DepthAnything/Depth-Anything-V2 external/Depth-Anything-V2\n"
        "  pip install -r external/Depth-Anything-V2/metric_depth/requirements.txt\n"
        "The file external/Depth-Anything-V2/metric_depth/depth_anything_v2/dpt.py "
        "must exist."
    )


def load_depth_anything_model(
    encoder: str = "vits",
    checkpoint_path: str = "checkpoints/pretrained/depth_anything_v2_metric_hypersim_vits.pth",
    max_depth: float = 10.0,
    device: str = "cuda",
):
    """Create Depth Anything V2, load pretrained weights and move it to device."""
    if encoder not in model_configs:
        available_encoders = ", ".join(model_configs.keys())
        raise ValueError(
            f"Unknown encoder: {encoder}. Available encoders: {available_encoders}"
        )

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Pretrained checkpoint was not found.\n"
            f"Expected file: {checkpoint_path}\n"
            "Download the correct Depth Anything V2 metric checkpoint and place it "
            "in checkpoints/pretrained/."
        )

    DepthAnythingV2 = import_depth_anything_v2()

    config = model_configs[encoder].copy()
    config["max_depth"] = max_depth

    model = DepthAnythingV2(**config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint)

    model = model.to(device)

    return model


def select_device(device_preference: str = "auto"):
    """Select CUDA, MPS or CPU according to config and local availability"""
    device_preference = str(device_preference).lower()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if device_preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if has_mps:
            return "mps"
        return "cpu"

    if device_preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but torch.cuda is not available.")

    if device_preference == "mps" and not has_mps:
        raise RuntimeError(
            "Config requested MPS, but torch.backends.mps is not available."
        )

    if device_preference not in ("cuda", "mps", "cpu"):
        raise ValueError("device must be one of: auto, cuda, mps, cpu")

    return device_preference


def get_depth_head_prefixes(model):
    """Return parameter prefixes that belong to the depth decoder/head."""
    prefixes = []
    for name, _module in model.named_modules():
        last_name = name.split(".")[-1]
        if "head" in last_name or "decoder" in last_name:
            prefixes.append(f"{name}.")

    return tuple(prefixes)


def get_encoder_prefixes(model):
    """Return parameter prefixes that belong to the Depth Anything encoder."""
    if hasattr(model, "pretrained"):
        return ("pretrained.",)

    prefixes = []
    for name, _module in model.named_modules():
        last_name = name.split(".")[-1]
        if last_name in ("encoder", "backbone"):
            prefixes.append(f"{name}.")

    return tuple(prefixes)


def is_depth_head_parameter(name, head_prefixes):
    """Check whether a named parameter belongs to the depth head/decoder."""
    return name.startswith(head_prefixes)


def is_encoder_parameter(name, encoder_prefixes):
    """Check whether a named parameter belongs to the encoder/backbone."""
    return name.startswith(encoder_prefixes)


def freeze_all_parameters(model):
    """Freeze the full model."""
    for parameter in model.parameters():
        parameter.requires_grad = False


def unfreeze_parameters_by_prefix(model, prefixes):
    """Unfreeze named parameters that match any of the given prefixes."""
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad = True


def unfreeze_last_encoder_blocks(model, last_blocks):
    """Unfreeze the last N transformer blocks of Depth Anything's encoder."""
    if last_blocks <= 0:
        return []

    if not hasattr(model, "pretrained") or not hasattr(model.pretrained, "blocks"):
        raise ValueError(
            "partial fine-tuning expects model.pretrained.blocks. "
            "This matches the official Depth Anything V2 DINOv2 encoder."
        )

    blocks = model.pretrained.blocks
    total_blocks = len(blocks)
    first_unfrozen = max(total_blocks - last_blocks, 0)
    unfrozen_prefixes = []

    for block_index in range(first_unfrozen, total_blocks):
        prefix = f"pretrained.blocks.{block_index}."
        unfrozen_prefixes.append(prefix)
        unfreeze_parameters_by_prefix(model, (prefix,))

    return unfrozen_prefixes


def get_parent_module(model, module_name):
    """Return parent module and child attribute name for a dotted module path."""
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)

    return parent, parts[-1]


def apply_lora_to_encoder(model, config, encoder_prefixes):
    """Replace selected encoder Linear layers with LoRA adapters."""
    rank = int(config.get("lora_rank", 8))
    alpha = float(config.get("lora_alpha", 16.0))
    dropout = float(config.get("lora_dropout", 0.0))
    target_modules = tuple(config.get("lora_target_modules", ["qkv", "proj"]))

    if rank <= 0:
        raise ValueError("lora_rank must be a positive integer.")

    replacements = []
    modules = list(model.named_modules())

    for module_name, module in modules:
        if not module_name.startswith(encoder_prefixes):
            continue
        if not isinstance(module, nn.Linear):
            continue

        last_name = module_name.split(".")[-1]
        if last_name not in target_modules:
            continue

        parent, child_name = get_parent_module(model, module_name)
        setattr(
            parent,
            child_name,
            LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout),
        )
        replacements.append(module_name)

    if not replacements:
        raise ValueError(
            "LoRA did not match any encoder Linear layers. "
            "Check lora_target_modules and Depth Anything V2 module names."
        )

    return replacements


def count_parameters(model):
    """Count total and trainable parameters."""
    total = 0
    trainable = 0

    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def configure_finetuning(model, config):
    """Apply a fine-tuning strategy and return optimizer parameter groups."""
    strategy = config.get("finetune_strategy", "full")
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    head_prefixes = get_depth_head_prefixes(model)
    encoder_prefixes = get_encoder_prefixes(model)
    unfrozen_encoder_prefixes = []
    lora_replaced_modules = []

    if not head_prefixes:
        raise ValueError(
            "Could not find a depth head/decoder module. "
            "Expected module name containing 'head' or 'decoder'."
        )

    if not encoder_prefixes:
        raise ValueError(
            "Could not find an encoder/backbone module. "
            "Expected Depth Anything V2 to expose model.pretrained."
        )

    if strategy == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
        parameter_groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "name": "all",
            }
        ]

    elif strategy == "head_only":
        freeze_all_parameters(model)
        unfreeze_parameters_by_prefix(model, head_prefixes)
        parameter_groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "name": "head",
            }
        ]

    elif strategy == "partial":
        freeze_all_parameters(model)
        unfreeze_parameters_by_prefix(model, head_prefixes)
        unfrozen_encoder_prefixes = unfreeze_last_encoder_blocks(
            model,
            int(config.get("unfreeze_last_blocks", 2)),
        )
        parameter_groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "name": "partial",
            }
        ]

    elif strategy == "layerwise_lr":
        for parameter in model.parameters():
            parameter.requires_grad = True

        encoder_lr = float(config.get("encoder_learning_rate", learning_rate * 0.1))
        head_lr = float(config.get("head_learning_rate", learning_rate))
        encoder_params = []
        head_params = []
        other_params = []

        for name, parameter in model.named_parameters():
            if is_depth_head_parameter(name, head_prefixes):
                head_params.append(parameter)
            elif is_encoder_parameter(name, encoder_prefixes):
                encoder_params.append(parameter)
            else:
                other_params.append(parameter)

        parameter_groups = [
            {
                "params": encoder_params,
                "lr": encoder_lr,
                "weight_decay": weight_decay,
                "name": "encoder",
            },
            {
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "head",
            },
        ]
        if other_params:
            parameter_groups.append(
                {
                    "params": other_params,
                    "lr": head_lr,
                    "weight_decay": weight_decay,
                    "name": "other",
                }
            )

    elif strategy == "lora":
        freeze_all_parameters(model)
        lora_replaced_modules = apply_lora_to_encoder(model, config, encoder_prefixes)
        train_head = bool(config.get("lora_train_head", True))
        if train_head:
            unfreeze_parameters_by_prefix(model, head_prefixes)

        lora_params = []
        head_params = []

        for name, parameter in model.named_parameters():
            if "lora_down." in name or "lora_up." in name:
                parameter.requires_grad = True
                lora_params.append(parameter)
            elif train_head and is_depth_head_parameter(name, head_prefixes):
                parameter.requires_grad = True
                head_params.append(parameter)

        lora_lr = float(config.get("lora_learning_rate", learning_rate))
        head_lr = float(config.get("head_learning_rate", learning_rate))
        parameter_groups = [
            {
                "params": lora_params,
                "lr": lora_lr,
                "weight_decay": weight_decay,
                "name": "lora",
            },
            {
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "head",
            },
        ]

    else:
        raise ValueError(
            "finetune_strategy must be one of: "
            "full, head_only, partial, layerwise_lr, lora"
        )

    parameter_groups = [group for group in parameter_groups if group["params"]]
    parameter_counts = count_parameters(model)

    return {
        "strategy": strategy,
        "parameter_groups": parameter_groups,
        "parameter_counts": parameter_counts,
        "head_prefixes": head_prefixes,
        "encoder_prefixes": encoder_prefixes,
        "unfrozen_encoder_prefixes": unfrozen_encoder_prefixes,
        "lora_replaced_modules": lora_replaced_modules,
    }
