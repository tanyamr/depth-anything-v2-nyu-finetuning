"""Predict depth map for one RGB image.

Example:
    python predict.py \
        --config configs/train_vits.yaml \
        --checkpoint outputs/experiments/vits_nyu/checkpoints/best_absrel.pth \
        --image path/to/image.jpg \
        --output outputs/predictions/result.png
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
import torchvision.transforms.functional as TF

from model import configure_finetuning, load_depth_anything_model, select_device


def load_config(config_path):
    """Read YAML config file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def prepare_prediction(pred_depth):
    """Convert model output to shape [B, 1, H, W]."""
    if pred_depth.ndim == 3:
        pred_depth = pred_depth.unsqueeze(1)
    elif pred_depth.ndim != 4:
        raise ValueError(f"Unexpected prediction shape: {pred_depth.shape}")

    return pred_depth


def load_trained_checkpoint(model, checkpoint_path, device):
    """Load fine-tuned checkpoint weights into the model."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Also supports plain state_dict checkpoints.
        model.load_state_dict(checkpoint)


def load_image(image_path, image_size):
    """Load and preprocess one RGB image."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    # Keep original size so we can save prediction at that size.
    original_size = image.size

    image = image.resize((image_size, image_size), Image.BILINEAR)
    image = TF.to_tensor(image)
    image = TF.normalize(
        image,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    return image.unsqueeze(0), original_size


def save_depth_outputs(depth, output_path, max_depth):
    """Save colored depth PNG with colorbar and raw depth NPY."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    npy_path = output_path.with_suffix(".npy")
    np.save(npy_path, depth.astype(np.float32))

    depth_vis = np.nan_to_num(depth, nan=0.0, posinf=max_depth, neginf=0.0)
    depth_vis = np.clip(depth_vis, 0.0, max_depth)
    depth_vis = depth_vis / max_depth

    fig = plt.figure(figsize=(6, 5), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 0.05], wspace=0.08)
    ax = fig.add_subplot(grid[0, 0])
    colorbar_axis = fig.add_subplot(grid[0, 1])

    depth_plot = ax.imshow(depth_vis * max_depth, cmap="turbo", vmin=0.0, vmax=max_depth)
    ax.set_title("Predicted depth map")
    ax.axis("off")

    colorbar = fig.colorbar(depth_plot, cax=colorbar_axis)
    colorbar.set_label("Depth, meters")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path, npy_path


@torch.no_grad()
def predict(model, image, original_size, config, device):
    """Run model prediction for one image."""
    model.eval()
    image = image.to(device)

    use_amp = bool(config["use_amp"]) and device == "cuda"

    with torch.cuda.amp.autocast(enabled=use_amp):
        pred_depth = model(image)
        pred_depth = prepare_prediction(pred_depth)

    pred_depth = pred_depth.squeeze(0).squeeze(0).detach().cpu().numpy()

    # Resize prediction back to the original image size.
    pred_depth_image = Image.fromarray(pred_depth.astype(np.float32))
    pred_depth_image = pred_depth_image.resize(original_size, Image.BILINEAR)

    return np.array(pred_depth_image, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--image", required=True, help="Path to input RGB image")
    parser.add_argument("--output", required=True, help="Path to output colored PNG")
    args = parser.parse_args()

    config = load_config(args.config)
    device = select_device(config.get("device", "auto"))

    print(f"Device: {device}")
    print(f"Image: {args.image}")
    print(f"Checkpoint: {args.checkpoint}")

    model = load_depth_anything_model(
        encoder=config["encoder"],
        checkpoint_path=config["checkpoint_path"],
        max_depth=config["max_depth"],
        device=device,
    )
    if config.get("finetune_strategy") == "lora":
        configure_finetuning(model, config)
    load_trained_checkpoint(model, args.checkpoint, device)

    image, original_size = load_image(args.image, config["image_size"])
    pred_depth = predict(model, image, original_size, config, device)

    png_path, npy_path = save_depth_outputs(
        pred_depth,
        args.output,
        max_depth=config["max_depth"],
    )

    print(f"Saved colored depth map: {png_path}")
    print(f"Saved raw depth map: {npy_path}")


if __name__ == "__main__":
    main()
