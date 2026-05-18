from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def instance_centroids(instance_map: np.ndarray) -> dict[int, tuple[float, float]]:
    centroids = {}
    for inst_id in np.unique(instance_map):
        if inst_id == 0:
            continue
        ys, xs = np.nonzero(instance_map == inst_id)
        if len(xs) > 0:
            centroids[int(inst_id)] = (float(xs.mean()), float(ys.mean()))
    return centroids


def match_instances_by_centroid(pred_map: np.ndarray, gt_map: np.ndarray, radius: float = 12.0) -> list[tuple[int, int, float]]:
    pred = instance_centroids(pred_map)
    gt = instance_centroids(gt_map)
    if not pred or not gt:
        return []
    pred_ids = list(pred.keys())
    gt_ids = list(gt.keys())
    dist = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
    for i, pid in enumerate(pred_ids):
        for j, gid in enumerate(gt_ids):
            dist[i, j] = np.hypot(pred[pid][0] - gt[gid][0], pred[pid][1] - gt[gid][1])
    rows, cols = linear_sum_assignment(dist)
    matches = []
    for r, c in zip(rows, cols):
        if dist[r, c] <= radius:
            matches.append((pred_ids[r], gt_ids[c], float(dist[r, c])))
    return matches


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union else 0.0
