from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.evaluation.matching import mask_iou, match_instances_by_centroid


def dice_score(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    pred_binary = pred_binary.astype(bool)
    gt_binary = gt_binary.astype(bool)
    denom = pred_binary.sum() + gt_binary.sum()
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(pred_binary, gt_binary).sum() / denom)


def detection_f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def panoptic_quality(pred_map: np.ndarray, gt_map: np.ndarray, matches: list[tuple[int, int, float]], iou_threshold: float = 0.5) -> float:
    iou_sum = 0.0
    tp = 0
    for pred_id, gt_id, _ in matches:
        iou = mask_iou(pred_map == pred_id, gt_map == gt_id)
        if iou >= iou_threshold:
            tp += 1
            iou_sum += iou
    fp = len([i for i in np.unique(pred_map) if i != 0]) - tp
    fn = len([i for i in np.unique(gt_map) if i != 0]) - tp
    denom = tp + 0.5 * fp + 0.5 * fn
    return float(iou_sum / denom) if denom else 0.0


def evaluate_segmentation_pair(pred_map: np.ndarray, gt_map: np.ndarray, pairing_radius: float = 12.0, iou_threshold: float = 0.5) -> dict[str, float]:
    matches = match_instances_by_centroid(pred_map, gt_map, radius=pairing_radius)
    pred_ids = set(int(i) for i in np.unique(pred_map) if i != 0)
    gt_ids = set(int(i) for i in np.unique(gt_map) if i != 0)
    tp = len(matches)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    return {
        "dice": dice_score(pred_map > 0, gt_map > 0),
        "detection_f1": detection_f1(tp, fp, fn),
        "pq": panoptic_quality(pred_map, gt_map, matches, iou_threshold=iou_threshold),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def classification_report_arrays(y_true, y_pred, labels, class_names=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    class_names = class_names or [str(x) for x in labels]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "macro": {"precision": float(p_macro), "recall": float(r_macro), "f1": float(f_macro)},
        "weighted": {"precision": float(p_weighted), "recall": float(r_weighted), "f1": float(f_weighted)},
        "per_class": {
            str(label): {
                "name": class_names[i],
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
    }
