"""Evaluate a fine-tuned Depth Anything V2 checkpoint on the test split.

Example:
    python test.py --config configs/train_vits.yaml \
        --checkpoint outputs/experiments/vits_nyu/checkpoints/best_absrel.pth
"""

import argparse
import json
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
from torch.utils.data import DataLoader

from dataset import NYUDepthDataset
from metrics import compute_all_metrics
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
    """Load fine-tuned weights into the model."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch")
    else:
        # This also supports plain state_dict files.
        model.load_state_dict(checkpoint)
        epoch = None

    return epoch


def create_test_loader(config):
    """Create dataloader for the test split."""
    image_size = (config["image_size"], config["image_size"])
    device = select_device(config.get("device", "auto"))

    test_dataset = NYUDepthDataset(
        split="test",
        image_size=image_size,
        min_depth=config["min_depth"],
        max_depth=config["max_depth"],
        split_file=config["test_list"],
    )

    return DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=device == "cuda",
    )


def denormalize_image(image):
    """Convert normalized tensor [3, H, W] back to uint8 RGB."""
    image = image.detach().cpu().float()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = torch.clamp(image, 0.0, 1.0)

    image = image.permute(1, 2, 0).numpy()
    return (image * 255).astype(np.uint8)


def depth_to_numpy(depth, max_depth):
    """Convert a depth tensor [H, W] to a clean NumPy array."""
    depth = depth.detach().cpu().float().numpy()
    depth = np.nan_to_num(depth, nan=0.0, posinf=max_depth, neginf=0.0)
    return np.clip(depth, 0.0, max_depth)


def save_visualization(image, gt_depth, pred_depth, output_path, max_depth):
    """Save RGB, ground truth depth and predicted depth with labels and legend."""
    rgb = denormalize_image(image)
    gt_depth = depth_to_numpy(gt_depth.squeeze(0), max_depth)
    pred_depth = depth_to_numpy(pred_depth.squeeze(0), max_depth)

    fig = plt.figure(figsize=(13, 4.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.045], wspace=0.08)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
    ]
    colorbar_axis = fig.add_subplot(grid[0, 3])

    axes[0].imshow(rgb)
    axes[0].set_title("RGB image")
    axes[0].axis("off")

    gt_plot = axes[1].imshow(gt_depth, cmap="turbo", vmin=0.0, vmax=max_depth)
    axes[1].set_title("Ground truth depth")
    axes[1].axis("off")

    pred_plot = axes[2].imshow(pred_depth, cmap="turbo", vmin=0.0, vmax=max_depth)
    axes[2].set_title("Predicted depth")
    axes[2].axis("off")

    colorbar = fig.colorbar(pred_plot, cax=colorbar_axis)
    colorbar.set_label("Depth, meters")

    fig.suptitle("NYU Depth V2 test example", fontsize=12)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def evaluate(model, test_loader, config, device, visualization_dir, num_visualizations):
    """Run evaluation and save a few visual examples."""
    model.eval()

    metric_sums = {
        "absrel": 0.0,
        "rmse": 0.0,
        "delta1": 0.0,
        "delta2": 0.0,
        "delta3": 0.0,
    }
    total_samples = 0
    saved_visualizations = 0
    use_amp = bool(config["use_amp"]) and device == "cuda"

    for image, depth, valid_mask in test_loader:
        image = image.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_depth = model(image)
            pred_depth = prepare_prediction(pred_depth)

        metrics = compute_all_metrics(pred_depth, depth, valid_mask)

        batch_size = image.size(0)
        total_samples += batch_size
        for name, value in metrics.items():
            metric_sums[name] += value * batch_size

        while saved_visualizations < num_visualizations:
            batch_index = saved_visualizations % batch_size
            if batch_index >= batch_size:
                break

            output_path = visualization_dir / f"test_{saved_visualizations:03d}.png"
            save_visualization(
                image[batch_index],
                depth[batch_index],
                pred_depth[batch_index],
                output_path,
                config["max_depth"],
            )
            saved_visualizations += 1

            if batch_index == batch_size - 1:
                break

    return {
        name: value / max(total_samples, 1) for name, value in metric_sums.items()
    }


def save_results(results_path, metrics, checkpoint_path, checkpoint_epoch):
    """Save test metrics to JSON."""
    results = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "test_metrics": metrics,
    }

    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


def print_metrics(metrics):
    """Print metrics in a readable format."""
    print("\nTest metrics")
    print(f"AbsRel : {metrics['absrel']:.6f}")
    print(f"RMSE   : {metrics['rmse']:.6f}")
    print(f"delta1 : {metrics['delta1']:.6f}")
    print(f"delta2 : {metrics['delta2']:.6f}")
    print(f"delta3 : {metrics['delta3']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument(
        "--experiment_name",
        default=None,
        help="Optional output folder name. Useful for pretrained baseline.",
    )
    parser.add_argument(
        "--num_visualizations",
        type=int,
        default=10,
        help="Number of test examples to save",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    experiment_name = args.experiment_name or config["experiment_name"]

    experiment_dir = Path(config["output_dir"]) / experiment_name
    visualization_dir = experiment_dir / "test_visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(config.get("device", "auto"))

    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    model = load_depth_anything_model(
        encoder=config["encoder"],
        checkpoint_path=config["checkpoint_path"],
        max_depth=config["max_depth"],
        device=device,
    )
    if config.get("finetune_strategy") == "lora":
        configure_finetuning(model, config)
    checkpoint_epoch = load_trained_checkpoint(model, checkpoint_path, device)

    test_loader = create_test_loader(config)
    metrics = evaluate(
        model,
        test_loader,
        config,
        device,
        visualization_dir,
        args.num_visualizations,
    )

    results_path = experiment_dir / "test_results.json"
    save_results(results_path, metrics, checkpoint_path, checkpoint_epoch)

    print_metrics(metrics)
    print(f"\nSaved results to: {results_path}")
    print(f"Saved visualizations to: {visualization_dir}")


if __name__ == "__main__":
    main()
