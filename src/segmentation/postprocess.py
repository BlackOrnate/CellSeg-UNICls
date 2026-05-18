from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import binary_dilation
from skimage.measure import label as cc_label
from skimage.morphology import dilation, disk


@dataclass
class InstanceInfo:
    bbox: list[list[int]]
    centroid: list[float]
    contour: list[list[int]]
    type: int = 0
    type_prob: float = 0.0


def get_bounding_box(mask: np.ndarray) -> list[int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(rmin), int(rmax + 1), int(cmin), int(cmax + 1)]


def calculate_instance_info(instance_map: np.ndarray, type_map: np.ndarray | None = None) -> dict[int, dict]:
    info: dict[int, dict] = {}
    for inst_id in np.unique(instance_map):
        if inst_id == 0:
            continue
        mask = instance_map == inst_id
        rmin, rmax, cmin, cmax = get_bounding_box(mask)
        crop = mask[rmin:rmax, cmin:cmax].astype(np.uint8)
        moments = cv2.moments(crop)
        if abs(moments["m00"]) < 1e-8:
            cx = (cmin + cmax) / 2.0
            cy = (rmin + rmax) / 2.0
        else:
            cx = moments["m10"] / moments["m00"] + cmin
            cy = moments["m01"] / moments["m00"] + rmin

        contours, _ = cv2.findContours(crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_list: list[list[int]] = []
        if contours:
            contour = max(contours, key=cv2.contourArea).squeeze(1)
            if contour.ndim == 2:
                contour[:, 0] += cmin
                contour[:, 1] += rmin
                contour_list = contour.astype(int).tolist()

        inst_type = 0
        type_prob = 0.0
        if type_map is not None:
            vals, counts = np.unique(type_map[mask], return_counts=True)
            order = np.argsort(-counts)
            for idx in order:
                if int(vals[idx]) != 0:
                    inst_type = int(vals[idx])
                    type_prob = float(counts[idx] / max(mask.sum(), 1))
                    break

        info[int(inst_id)] = {
            "bbox": [[rmin, cmin], [rmax, cmax]],
            "centroid": [float(cx), float(cy)],
            "contour": contour_list,
            "type": inst_type,
            "type_prob": type_prob,
        }
    return info


def morphology_aware_merge(
    instance_map: np.ndarray,
    instance_info: dict[int, dict],
    target_types: set[int] | None = None,
    radius: int = 1,
) -> tuple[np.ndarray, dict[int, dict]]:
    """Merge nearby same-class fragments after a conservative dilation step.

    The thesis analyzes this as an optional ablation, mainly for neuron and blood vessel classes.
    By default, target type IDs are `{2, 5}`.
    """
    target_types = target_types or {2, 5}
    merged = instance_map.copy().astype(np.int32)

    for class_id in sorted(target_types):
        class_mask = np.zeros_like(instance_map, dtype=np.int32)
        for inst_id, meta in instance_info.items():
            if int(meta.get("type", 0)) == class_id:
                class_mask[instance_map == int(inst_id)] = int(inst_id)
        if class_mask.max() == 0:
            continue
        dilated = dilation(class_mask > 0, footprint=disk(radius)).astype(np.uint8)
        cc = cc_label(dilated)
        for comp_id in np.unique(cc):
            if comp_id == 0:
                continue
            ids = np.unique(class_mask[cc == comp_id])
            ids = ids[ids != 0]
            if len(ids) <= 1:
                continue
            new_id = int(merged.max()) + 1
            merged[np.isin(merged, ids)] = new_id

    # Re-index to consecutive IDs.
    relabeled = np.zeros_like(merged, dtype=np.int32)
    for new_id, old_id in enumerate([i for i in np.unique(merged) if i != 0], start=1):
        relabeled[merged == old_id] = new_id
    new_info = calculate_instance_info(relabeled)

    # Inherit type by majority overlap from old instance map.
    for new_id in new_info:
        mask = relabeled == new_id
        old_ids, counts = np.unique(instance_map[mask], return_counts=True)
        valid = old_ids != 0
        if valid.any():
            best = int(old_ids[valid][np.argmax(counts[valid])])
            new_info[new_id]["type"] = int(instance_info.get(best, {}).get("type", 0))
            new_info[new_id]["type_prob"] = float(instance_info.get(best, {}).get("type_prob", 0.0))
    return relabeled, new_info


def type_map_from_instance_info(instance_map: np.ndarray, instance_info: dict[int, dict]) -> np.ndarray:
    type_map = np.zeros_like(instance_map, dtype=np.int32)
    for inst_id, meta in instance_info.items():
        type_map[instance_map == int(inst_id)] = int(meta.get("type", 0))
    return type_map
