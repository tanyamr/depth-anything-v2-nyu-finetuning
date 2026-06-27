""" loss functions """

import torch


def masked_l1_loss(pred_depth, gt_depth, valid_mask):
    """Calculate L1 loss only for valid depth pixels."""
    pred_valid = pred_depth[valid_mask]
    gt_valid = gt_depth[valid_mask]

    # If the batch has no valid pixels, return zero loss on the right device.
    if pred_valid.numel() == 0:
        return pred_depth.sum() * 0.0

    return torch.mean(torch.abs(pred_valid - gt_valid))


def silog_loss(pred_depth, gt_depth, valid_mask):
    """Calculate scale-invariant logarithmic loss.

    This loss compares log-depth values and is useful when relative depth
    structure matters. We clamp values before log to avoid log(0).
    """
    pred_valid = pred_depth[valid_mask]
    gt_valid = gt_depth[valid_mask]

    if pred_valid.numel() == 0:
        return pred_depth.sum() * 0.0

    eps = 1e-6
    pred_valid = torch.clamp(pred_valid, min=eps)
    gt_valid = torch.clamp(gt_valid, min=eps)

    log_diff = torch.log(pred_valid) - torch.log(gt_valid)

    # SiLog: mean(d^2) - mean(d)^2
    loss = torch.mean(log_diff**2) - torch.mean(log_diff) ** 2

    # Numerical errors can make the value slightly negative.
    return torch.sqrt(torch.clamp(loss, min=eps))


def depth_loss(
    pred_depth,
    gt_depth,
    valid_mask,
    l1_weight=1.0,
    silog_weight=0.5,
):
    """Combine masked L1 loss and SiLog loss."""
    l1 = masked_l1_loss(pred_depth, gt_depth, valid_mask)
    silog = silog_loss(pred_depth, gt_depth, valid_mask)

    return l1_weight * l1 + silog_weight * silog
