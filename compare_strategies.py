"""Summarize fine-tuning strategy experiment results.

Example:
    python compare_strategies.py \
        --experiments vitb_nyu vitb_head_only_mps vitb_partial_mps vitb_layerwise_lr_mps vitb_lora_mps
"""

import argparse
import csv
import json
from pathlib import Path

import yaml


DEFAULT_EXPERIMENTS = [
    "vitb_nyu",
    "vitb_head_only_mps",
    "vitb_partial_mps",
    "vitb_layerwise_lr_enc1e-7_mps",
    "vitb_layerwise_lr_mps",
    "vitb_layerwise_lr_enc1e-6_mps",
    "vitb_partial_4blocks_mps",
    "vitb_lora_rank4_mps",
    "vitb_lora_mps",
    "vitb_lora_rank16_mps",
]


def load_json(path):
    """Load JSON if it exists, otherwise return an empty dict."""
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_yaml(path):
    """Load YAML if it exists, otherwise return an empty dict."""
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_metrics_summary(path):
    """Read best validation metrics from metrics.csv if results.json is missing."""
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        return {}

    best_absrel = min(rows, key=lambda row: float(row["val_absrel"]))
    best_rmse = min(rows, key=lambda row: float(row["val_rmse"]))

    return {
        "best_absrel": float(best_absrel["val_absrel"]),
        "best_rmse": float(best_rmse["val_rmse"]),
        "best_epoch_absrel": int(best_absrel["epoch"]),
        "best_epoch_rmse": int(best_rmse["epoch"]),
    }


def compute_parameter_counts(config_path):
    """Recompute parameter counts when old results.json lacks this metadata."""
    if not config_path.exists():
        return {}

    try:
        from model import configure_finetuning, load_depth_anything_model
        from train import load_config
    except ImportError:
        return {}

    config = load_config(config_path)
    config["device"] = "cpu"
    model = load_depth_anything_model(
        encoder=config["encoder"],
        checkpoint_path=config["checkpoint_path"],
        max_depth=config["max_depth"],
        device="cpu",
    )
    finetuning_info = configure_finetuning(model, config)
    return finetuning_info["parameter_counts"]


def collect_row(experiment_dir):
    """Collect validation and test metrics for one experiment."""
    results = load_json(experiment_dir / "results.json")
    metrics_summary = load_metrics_summary(experiment_dir / "logs" / "metrics.csv")
    test_results = load_json(experiment_dir / "test_results.json")
    config_path = experiment_dir / "config.yaml"
    config = load_yaml(config_path)
    finetuning = results.get("finetuning", {})
    parameter_counts = finetuning.get("parameter_counts", {})
    if not parameter_counts:
        parameter_counts = compute_parameter_counts(config_path)

    test_metrics = test_results.get("test_metrics", {})
    strategy = finetuning.get("strategy") or config.get("finetune_strategy", "full")

    return {
        "experiment": experiment_dir.name,
        "strategy": strategy,
        "total_params": parameter_counts.get("total", ""),
        "trainable_params": parameter_counts.get("trainable", ""),
        "frozen_params": parameter_counts.get("frozen", ""),
        "best_val_absrel": results.get(
            "best_absrel",
            metrics_summary.get("best_absrel", ""),
        ),
        "best_val_rmse": results.get(
            "best_rmse",
            metrics_summary.get("best_rmse", ""),
        ),
        "best_epoch_absrel": results.get(
            "best_epoch_absrel",
            metrics_summary.get("best_epoch_absrel", ""),
        ),
        "test_absrel": test_metrics.get("absrel", ""),
        "test_rmse": test_metrics.get("rmse", ""),
        "test_delta1": test_metrics.get("delta1", ""),
        "test_delta2": test_metrics.get("delta2", ""),
        "test_delta3": test_metrics.get("delta3", ""),
    }


def to_float(value):
    """Convert a table value to float, returning None for missing values."""
    if value == "" or value is None:
        return None
    return float(value)


def enrich_efficiency_metrics(rows):
    """Add efficiency-oriented columns derived from quality and parameter counts."""
    trainable_values = [
        int(row["trainable_params"])
        for row in rows
        if row.get("trainable_params") not in ("", None)
    ]
    full_trainable = next(
        (
            int(row["trainable_params"])
            for row in rows
            if row.get("strategy") == "full"
            and row.get("trainable_params") not in ("", None)
        ),
        max(trainable_values) if trainable_values else None,
    )

    test_absrels = [
        to_float(row.get("test_absrel"))
        for row in rows
        if to_float(row.get("test_absrel")) is not None
    ]
    test_rmses = [
        to_float(row.get("test_rmse"))
        for row in rows
        if to_float(row.get("test_rmse")) is not None
    ]
    best_absrel = min(test_absrels) if test_absrels else None
    best_rmse = min(test_rmses) if test_rmses else None

    for row in rows:
        total_params = to_float(row.get("total_params"))
        trainable_params = to_float(row.get("trainable_params"))
        test_absrel = to_float(row.get("test_absrel"))
        test_rmse = to_float(row.get("test_rmse"))

        if trainable_params is not None:
            row["trainable_mparams"] = trainable_params / 1_000_000
        else:
            row["trainable_mparams"] = ""

        if trainable_params is not None and total_params:
            row["trainable_percent"] = trainable_params / total_params * 100.0
        else:
            row["trainable_percent"] = ""

        if trainable_params is not None and full_trainable:
            row["param_reduction_vs_full_percent"] = (
                1.0 - trainable_params / full_trainable
            ) * 100.0
        else:
            row["param_reduction_vs_full_percent"] = ""

        if test_absrel is not None and best_absrel is not None:
            row["absrel_gap_vs_best"] = test_absrel - best_absrel
        else:
            row["absrel_gap_vs_best"] = ""

        if test_rmse is not None and best_rmse is not None:
            row["rmse_gap_vs_best"] = test_rmse - best_rmse
        else:
            row["rmse_gap_vs_best"] = ""

    return rows


def write_csv(path, rows):
    """Write comparison table as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value):
    """Format table values for Markdown."""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(path, rows):
    """Write comparison table as Markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())

    with path.open("w", encoding="utf-8") as file:
        file.write("# Fine-tuning strategy comparison\n\n")
        file.write("| " + " | ".join(columns) + " |\n")
        file.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            values = [format_value(row[column]) for column in columns]
            file.write("| " + " | ".join(values) + " |\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help="Experiment folder names under outputs/experiments",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/report",
        help="Directory for comparison CSV and Markdown files",
    )
    parser.add_argument(
        "--experiments_dir",
        default="outputs/experiments",
        help="Directory containing experiment folders",
    )
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    rows = [
        collect_row(experiments_dir / experiment_name)
        for experiment_name in args.experiments
    ]
    rows = enrich_efficiency_metrics(rows)

    output_dir = Path(args.output_dir)
    csv_path = output_dir / "fine_tuning_strategy_comparison.csv"
    md_path = output_dir / "fine_tuning_strategy_comparison.md"

    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
