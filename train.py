"""Train Depth Anything V2 on NYU Depth V2.

Example:
    python train.py --config configs/train_vits.yaml
"""

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import NYUDepthDataset
from losses import depth_loss
from metrics import compute_all_metrics
from model import configure_finetuning, load_depth_anything_model, select_device


def load_config(config_path):
    """Read YAML config file."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return normalize_config(config)


def normalize_config(config):
    """Convert YAML values to the types expected by training code."""
    defaults = {
        "device": "auto",
        "finetune_strategy": "full",
        "unfreeze_last_blocks": 2,
        "encoder_learning_rate": None,
        "head_learning_rate": None,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "lora_dropout": 0.0,
        "lora_learning_rate": None,
        "lora_target_modules": ["qkv", "proj"],
        "lora_train_head": True,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)

    int_keys = [
        "image_size",
        "epochs",
        "batch_size",
        "num_workers",
        "save_every",
        "visualize_every",
    ]
    float_keys = [
        "learning_rate",
        "weight_decay",
        "min_depth",
        "max_depth",
        "l1_weight",
        "silog_weight",
    ]
    optional_float_keys = [
        "encoder_learning_rate",
        "head_learning_rate",
        "lora_alpha",
        "lora_dropout",
        "lora_learning_rate",
    ]

    for key in int_keys:
        config[key] = int(config[key])

    for key in float_keys:
        config[key] = float(config[key])

    for key in optional_float_keys:
        if config[key] is not None:
            config[key] = float(config[key])

    config["use_amp"] = bool(config["use_amp"])
    config["lora_train_head"] = bool(config["lora_train_head"])
    config["unfreeze_last_blocks"] = int(config["unfreeze_last_blocks"])
    config["lora_rank"] = int(config["lora_rank"])

    return config


def create_experiment_dirs(config, config_path):
    """Create output folders and save a copy of the config."""
    experiment_dir = Path(config["output_dir"]) / config["experiment_name"]
    checkpoints_dir = experiment_dir / "checkpoints"
    logs_dir = experiment_dir / "logs"
    tensorboard_dir = logs_dir / "tensorboard"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(config_path, experiment_dir / "config.yaml")

    return experiment_dir, checkpoints_dir, logs_dir, tensorboard_dir


def create_dataloaders(config, device):
    """Create train and validation dataloaders."""
    image_size = (config["image_size"], config["image_size"])
    pin_memory = device == "cuda"

    train_dataset = NYUDepthDataset(
        split="train",
        image_size=image_size,
        min_depth=config["min_depth"],
        max_depth=config["max_depth"],
        split_file=config["train_list"],
    )
    val_dataset = NYUDepthDataset(
        split="val",
        image_size=image_size,
        min_depth=config["min_depth"],
        max_depth=config["max_depth"],
        split_file=config["val_list"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def prepare_prediction(pred_depth):
    """Convert model output to shape [B, 1, H, W]."""
    if pred_depth.ndim == 3:
        pred_depth = pred_depth.unsqueeze(1)
    elif pred_depth.ndim != 4:
        raise ValueError(f"Unexpected prediction shape: {pred_depth.shape}")

    return pred_depth


def train_one_epoch(model, train_loader, optimizer, scaler, config, device, use_amp):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for image, depth, valid_mask in train_loader:
        image = image.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_depth = model(image)
            pred_depth = prepare_prediction(pred_depth)
            loss = depth_loss(
                pred_depth,
                depth,
                valid_mask,
                l1_weight=config["l1_weight"],
                silog_weight=config["silog_weight"],
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = image.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def validate(model, val_loader, config, device, use_amp):
    """Evaluate model on validation split."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    metric_sums = {
        "absrel": 0.0,
        "rmse": 0.0,
        "delta1": 0.0,
        "delta2": 0.0,
        "delta3": 0.0,
    }

    for image, depth, valid_mask in val_loader:
        image = image.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_depth = model(image)
            pred_depth = prepare_prediction(pred_depth)
            loss = depth_loss(
                pred_depth,
                depth,
                valid_mask,
                l1_weight=config["l1_weight"],
                silog_weight=config["silog_weight"],
            )

        metrics = compute_all_metrics(pred_depth, depth, valid_mask)

        batch_size = image.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        for name, value in metrics.items():
            metric_sums[name] += value * batch_size

    val_loss = total_loss / max(total_samples, 1)
    val_metrics = {
        name: value / max(total_samples, 1) for name, value in metric_sums.items()
    }

    return val_loss, val_metrics


def save_checkpoint(
    checkpoint_path,
    epoch,
    model,
    optimizer,
    best_absrel,
    best_rmse,
    config,
):
    """Save a training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_absrel": best_absrel,
        "best_rmse": best_rmse,
        "config": config,
    }
    torch.save(checkpoint, checkpoint_path)


def create_metrics_csv(csv_path):
    """Create CSV file and write its header."""
    columns = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_absrel",
        "val_rmse",
        "val_delta1",
        "val_delta2",
        "val_delta3",
        "learning_rate",
        "encoder_learning_rate",
        "head_learning_rate",
        "lora_learning_rate",
        "epoch_time_sec",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

    return columns


def append_metrics_csv(csv_path, columns, row):
    """Append one epoch row to metrics.csv."""
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writerow(row)


def write_tensorboard(writer, epoch, train_loss, val_loss, val_metrics, learning_rates):
    """Write numeric values to TensorBoard."""
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Metrics/AbsRel", val_metrics["absrel"], epoch)
    writer.add_scalar("Metrics/RMSE", val_metrics["rmse"], epoch)
    writer.add_scalar("Metrics/Delta1", val_metrics["delta1"], epoch)
    writer.add_scalar("Metrics/Delta2", val_metrics["delta2"], epoch)
    writer.add_scalar("Metrics/Delta3", val_metrics["delta3"], epoch)
    writer.add_scalar("LearningRate/main", learning_rates["main"], epoch)
    if learning_rates["encoder"] is not None:
        writer.add_scalar("LearningRate/encoder", learning_rates["encoder"], epoch)
    if learning_rates["head"] is not None:
        writer.add_scalar("LearningRate/head", learning_rates["head"], epoch)
    if learning_rates["lora"] is not None:
        writer.add_scalar("LearningRate/lora", learning_rates["lora"], epoch)


def get_learning_rates(optimizer):
    """Summarize optimizer learning rates for logs."""
    named_lrs = {group.get("name", f"group_{index}"): group["lr"] for index, group in enumerate(optimizer.param_groups)}
    lrs = [group["lr"] for group in optimizer.param_groups]

    return {
        "main": max(lrs) if lrs else None,
        "encoder": named_lrs.get("encoder"),
        "head": named_lrs.get("head"),
        "lora": named_lrs.get("lora"),
    }


def format_parameter_count(count):
    """Format a parameter count as a compact human-readable string."""
    return f"{count:,}".replace(",", " ")


def save_results(
    experiment_dir,
    best_absrel,
    best_rmse,
    best_epoch_absrel,
    best_epoch_rmse,
    final_val_metrics,
    total_training_time,
    finetuning_info,
):
    """Save final training summary."""
    results = {
        "best_absrel": best_absrel,
        "best_rmse": best_rmse,
        "best_epoch_absrel": best_epoch_absrel,
        "best_epoch_rmse": best_epoch_rmse,
        "final_val_metrics": final_val_metrics,
        "total_training_time": total_training_time,
        "finetuning": {
            "strategy": finetuning_info["strategy"],
            "parameter_counts": finetuning_info["parameter_counts"],
            "head_prefixes": list(finetuning_info["head_prefixes"]),
            "encoder_prefixes": list(finetuning_info["encoder_prefixes"]),
            "unfrozen_encoder_prefixes": finetuning_info[
                "unfrozen_encoder_prefixes"
            ],
            "lora_replaced_modules": finetuning_info["lora_replaced_modules"],
            "lora_rank": config.get("lora_rank"),
            "lora_alpha": config.get("lora_alpha"),
            "lora_dropout": config.get("lora_dropout"),
            "lora_target_modules": config.get("lora_target_modules"),
            "lora_train_head": config.get("lora_train_head"),
        },
    }

    with (experiment_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    experiment_dir, checkpoints_dir, logs_dir, tensorboard_dir = create_experiment_dirs(
        config, config_path
    )

    device = select_device(config["device"])
    use_amp = bool(config["use_amp"]) and device == "cuda"

    print(f"Experiment: {config['experiment_name']}")
    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Fine-tuning strategy: {config['finetune_strategy']}")

    train_loader, val_loader = create_dataloaders(config, device)

    model = load_depth_anything_model(
        encoder=config["encoder"],
        checkpoint_path=config["checkpoint_path"],
        max_depth=config["max_depth"],
        device=device,
    )

    finetuning_info = configure_finetuning(model, config)
    parameter_counts = finetuning_info["parameter_counts"]
    print(
        "Parameters: "
        f"trainable={format_parameter_count(parameter_counts['trainable'])} | "
        f"frozen={format_parameter_count(parameter_counts['frozen'])} | "
        f"total={format_parameter_count(parameter_counts['total'])}"
    )
    for group in finetuning_info["parameter_groups"]:
        print(
            f"Optimizer group '{group['name']}': "
            f"params={len(group['params'])} | lr={group['lr']}"
        )

    if parameter_counts["trainable"] == 0:
        raise ValueError("No trainable parameters after applying fine-tuning strategy.")

    optimizer = torch.optim.AdamW(finetuning_info["parameter_groups"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    csv_path = logs_dir / "metrics.csv"
    csv_columns = create_metrics_csv(csv_path)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    best_absrel = float("inf")
    best_rmse = float("inf")
    best_epoch_absrel = 0
    best_epoch_rmse = 0
    final_val_metrics = None
    training_start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        epoch_start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, config, device, use_amp
        )
        val_loss, val_metrics = validate(model, val_loader, config, device, use_amp)
        final_val_metrics = val_metrics

        epoch_time = time.time() - epoch_start_time
        learning_rates = get_learning_rates(optimizer)

        print(
            f"Epoch {epoch:03d}/{config['epochs']} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"absrel={val_metrics['absrel']:.4f} | "
            f"rmse={val_metrics['rmse']:.4f}"
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_absrel": val_metrics["absrel"],
            "val_rmse": val_metrics["rmse"],
            "val_delta1": val_metrics["delta1"],
            "val_delta2": val_metrics["delta2"],
            "val_delta3": val_metrics["delta3"],
            "learning_rate": learning_rates["main"],
            "encoder_learning_rate": learning_rates["encoder"],
            "head_learning_rate": learning_rates["head"],
            "lora_learning_rate": learning_rates["lora"],
            "epoch_time_sec": epoch_time,
        }
        append_metrics_csv(csv_path, csv_columns, row)
        write_tensorboard(
            writer, epoch, train_loss, val_loss, val_metrics, learning_rates
        )

        save_checkpoint(
            checkpoints_dir / "last.pth",
            epoch,
            model,
            optimizer,
            best_absrel,
            best_rmse,
            config,
        )

        if val_metrics["absrel"] < best_absrel:
            best_absrel = val_metrics["absrel"]
            best_epoch_absrel = epoch
            save_checkpoint(
                checkpoints_dir / "best_absrel.pth",
                epoch,
                model,
                optimizer,
                best_absrel,
                best_rmse,
                config,
            )

        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            best_epoch_rmse = epoch
            save_checkpoint(
                checkpoints_dir / "best_rmse.pth",
                epoch,
                model,
                optimizer,
                best_absrel,
                best_rmse,
                config,
            )

        if epoch % config["save_every"] == 0:
            save_checkpoint(
                checkpoints_dir / f"epoch_{epoch:03d}.pth",
                epoch,
                model,
                optimizer,
                best_absrel,
                best_rmse,
                config,
            )

    total_training_time = time.time() - training_start_time
    writer.close()

    save_results(
        experiment_dir,
        best_absrel,
        best_rmse,
        best_epoch_absrel,
        best_epoch_rmse,
        final_val_metrics,
        total_training_time,
        finetuning_info,
    )

    print("Training finished.")
    print(f"Best AbsRel: {best_absrel:.4f} at epoch {best_epoch_absrel}")
    print(f"Best RMSE: {best_rmse:.4f} at epoch {best_epoch_rmse}")


if __name__ == "__main__":
    main()
