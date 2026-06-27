"""Metrics for monocular depth estimation."""

import torch


def _select_valid_pixels(pred_depth, gt_depth, valid_mask):
    """Return only pixels selected by valid_mask."""
    pred_valid = pred_depth[valid_mask]
    gt_valid = gt_depth[valid_mask]
    return pred_valid, gt_valid


def compute_absrel(pred_depth, gt_depth, valid_mask):
    """Compute Absolute Relative Error."""
    pred_valid, gt_valid = _select_valid_pixels(pred_depth, gt_depth, valid_mask)

    if pred_valid.numel() == 0:
        return 0.0

    eps = 1e-6
    gt_valid = torch.clamp(gt_valid, min=eps)
    absrel = torch.mean(torch.abs(pred_valid - gt_valid) / gt_valid)

    return float(absrel.item())


def compute_rmse(pred_depth, gt_depth, valid_mask):
    """Compute Root Mean Squared Error."""
    pred_valid, gt_valid = _select_valid_pixels(pred_depth, gt_depth, valid_mask)

    if pred_valid.numel() == 0:
        return 0.0

    rmse = torch.sqrt(torch.mean((pred_valid - gt_valid) ** 2))

    return float(rmse.item())


def compute_delta(pred_depth, gt_depth, valid_mask, threshold):
    """Compute delta accuracy for a selected threshold."""
    pred_valid, gt_valid = _select_valid_pixels(pred_depth, gt_depth, valid_mask)

    if pred_valid.numel() == 0:
        return 0.0

    eps = 1e-6
    pred_valid = torch.clamp(pred_valid, min=eps)
    gt_valid = torch.clamp(gt_valid, min=eps)

    ratio = torch.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    delta = torch.mean((ratio < threshold).float())

    return float(delta.item())


def compute_all_metrics(pred_depth, gt_depth, valid_mask):
    """Compute all depth estimation metrics."""
    return {
        "absrel": compute_absrel(pred_depth, gt_depth, valid_mask),
        "rmse": compute_rmse(pred_depth, gt_depth, valid_mask),
        "delta1": compute_delta(pred_depth, gt_depth, valid_mask, 1.25),
        "delta2": compute_delta(pred_depth, gt_depth, valid_mask, 1.25**2),
        "delta3": compute_delta(pred_depth, gt_depth, valid_mask, 1.25**3),
    }
