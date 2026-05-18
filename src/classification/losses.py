from __future__ import annotations

import torch
import torch.nn.functional as F


def map_labels_to_zero_based(labels: torch.Tensor, label_offset: int = 1, ignore_label: int = -1) -> torch.Tensor:
    """Map project type IDs to PyTorch class indices.

    Example: if semantic labels are 1..6 and `label_offset=1`, they become 0..5.
    Ignored labels remain `ignore_label`.
    """
    labels = labels.clone().long()
    valid = labels != ignore_label
    labels[valid] = labels[valid] - label_offset
    return labels


def classification_loss(logits: torch.Tensor, labels: torch.Tensor, label_offset: int = 1, ignore_label: int = -1):
    mapped = map_labels_to_zero_based(labels, label_offset=label_offset, ignore_label=ignore_label)
    return F.cross_entropy(logits, mapped, ignore_index=ignore_label)
