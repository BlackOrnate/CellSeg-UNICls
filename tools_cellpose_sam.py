import numpy
import numpy as np
from dataclasses import dataclass
import os
import csv
import tqdm
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_dilation
import torch.nn.functional as F
import torch
import cv2
from PIL import Image, ImageDraw
from torchvision import transforms
from skimage.color import rgba2rgb, label2rgb
from skimage.morphology import remove_small_objects, remove_small_holes, dilation, erosion, disk
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score, confusion_matrix
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import label as cc_label
from skimage.segmentation import find_boundaries
import logging
import re
from collections import defaultdict
import json
import numpy as np
from skimage.morphology import erosion, dilation, disk
from skimage.measure import label as cc_label, find_contours


def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def unpack_predictions(pred_inst_maps, image_names=None, uni_result_path=None, type_map_path=None, type_maps=None):
    predictions = {}
    predictions["nuclei_binary_map"] = [(mask > 0).astype(np.int32) for mask in pred_inst_maps]
    predictions["instance_map"] = pred_inst_maps
    predictions["instance_map"], predictions["instance_types"] = calculate_instances(pred_inst_maps, group="pred",
                                                                                     image_names=image_names,
                                                                                     uni_result_path=uni_result_path,
                                                                                     type_map_path=type_map_path,
                                                                                     type_maps=type_maps)
    predictions["instance_types_nuclei"] = generate_instance_nuclei_map(predictions["instance_map"],
                                                                        predictions["instance_types"])
    predictions["type_map"] = generate_type_map(predictions["instance_types"], pred_inst_maps[0].shape)
    return predictions


def unpack_predictions_for_only_image(pred_inst_maps, image_names=None, uni_result_path=None, type_map_path=None, ):
    predictions = {}
    predictions["instance_types"] = calculate_instances_for_only_image(pred_inst_maps,
                                                                       image_names=image_names,
                                                                       # uni_result_path=uni_result_path,
                                                                       type_map_path=type_map_path, )
    return predictions


def unpack_ground_truth(gt_inst_maps, image_names=None, type_map_path=None, type_maps=None):
    gt = {}
    gt["nuclei_binary_map"] = [(mask > 0).astype(np.int32) for mask in gt_inst_maps]
    gt["instance_map"] = gt_inst_maps
    gt["instance_map"], gt["instance_types"] = calculate_instances(gt_inst_maps, group="gt", image_names=image_names,
                                                                   type_map_path=type_map_path, type_maps=type_maps)
    gt["instance_types_nuclei"] = generate_instance_nuclei_map(gt["instance_map"], gt["instance_types"])
    # gt["type_map"] = generate_type_map(gt["instance_types"], gt_inst_maps[0].shape)
    return gt


def generate_type_map(instance_types: list, image_shape: tuple):
    if len(image_shape) == 3:
        h, w = image_shape[:2]
    else:
        h, w = image_shape

    type_maps = []

    for inst_dict in instance_types:
        type_map = np.zeros((h, w), dtype=np.int32)

        for cell_id, cell_info in inst_dict.items():
            contour = cell_info.get("contour", None)
            cell_type = cell_info.get("type", 0)

            if contour is None:
                continue

            contour = np.asarray(contour, dtype=np.int32)

            # OpenCV 需要 shape = (N,1,2)
            if contour.ndim == 2:
                contour = contour.reshape((-1, 1, 2))

            cv2.drawContours(
                image=type_map,
                contours=[contour],
                contourIdx=-1,
                color=int(cell_type),
                thickness=-1
            )

        type_maps.append(type_map)

    return type_maps


def generate_instance_nuclei_map(
        instance_maps: list, type_preds: list, num_nuclei_classes: int = 8
):
    instance_type_nuclei_maps = []

    for i in range(len(instance_maps)):
        instance_map = instance_maps[i]
        type_pred = type_preds[i]

        h, w = instance_map.shape

        instance_type_nuclei_map = np.zeros(
            (h, w, num_nuclei_classes),
            dtype=instance_map.dtype
        )

        for nuclei, spec in type_pred.items():
            nuclei_type = spec["type"]

            mask = (instance_map == nuclei)

            instance_type_nuclei_map[:, :, nuclei_type][mask] = nuclei

        instance_type_nuclei_maps.append(np.transpose(instance_type_nuclei_map, (2, 0, 1)))

    return instance_type_nuclei_maps


def calculate_instances(inst_maps, group, image_names, uni_result_path=None, type_map_path=None, type_maps=None,
                        use_dilation=True):
    new_inst_maps = []
    geojson_results = []
    for i, pred_inst in enumerate(inst_maps):
        inst_info_dict: dict[int, dict] = {}
        for inst_id in np.unique(pred_inst):
            if inst_id != 0:
                m = (pred_inst == inst_id)

                # bbox
                rmin, rmax, cmin, cmax = get_bounding_box(m)
                bbox = np.array([[rmin, cmin], [rmax, cmax]], dtype=np.int32)

                # centroid (x,y) using moments on cropped mask
                crop = m[rmin:rmax, cmin:cmax].astype(np.uint8)

                M = cv2.moments(crop)
                if abs(M["m00"]) < 1e-8:
                    cx = (cmin + cmax) / 2.0
                    cy = (rmin + rmax) / 2.0
                else:
                    cx = (M["m10"] / M["m00"]) + cmin
                    cy = (M["m01"] / M["m00"]) + rmin
                centroid = np.array([cx, cy], dtype=np.float32)  # (x,y)

                contours, _ = cv2.findContours(crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) == 0:
                    continue

                cnt = max(contours, key=cv2.contourArea).squeeze(1)  # (N,2) in crop coords

                if cnt.ndim != 2 or cnt.shape[0] < 3:
                    continue
                cnt[:, 0] += cmin  # x
                cnt[:, 1] += rmin  # y
                contour = cnt.astype(np.int32)

                inst_info_dict[int(inst_id)] = {
                    "bbox": bbox,
                    "centroid": centroid,  # (x,y)
                    "contour": contour,  # (x,y)
                    "type": 0,
                    "type_prob": 0
                }

        if type_maps and inst_maps[0].shape == type_maps[0].shape:
            if group == "pred":
                x = type_maps[i]
                # softmax
                exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # numerical stability
                softmax = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

                # argmax -> final class map
                pred_type = np.argmax(softmax, axis=-1)
            elif group == "gt":
                pred_type = type_maps[i]

            for inst_id in list(inst_info_dict.keys()):
                rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
                inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
                inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
                inst_map_crop = inst_map_crop == inst_id
                inst_type = inst_type_crop[inst_map_crop]

                type_list, type_pixels = np.unique(inst_type, return_counts=True)
                type_list = list(zip(type_list, type_pixels))
                type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
                inst_type = type_list[0][0]

                if inst_type == 0:  # ! pick the 2nd most dominant if exist
                    if len(type_list) > 1:
                        inst_type = type_list[1][0]
                type_dict = {v[0]: v[1] for v in type_list}
                type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
                inst_info_dict[inst_id]["type"] = int(inst_type)
                inst_info_dict[inst_id]["type_prob"] = float(type_prob)
        elif uni_result_path:
            with open(f"{uni_result_path}/{image_names[i]}.json", "r", encoding="utf-8") as f:
                pred_type_dict = json.load(f)
                for inst_id in list(inst_info_dict.keys()):
                    id = str(inst_id)
                    if id in pred_type_dict.keys():
                        if pred_type_dict[id]["type"] == 0:
                            print(123)
                        inst_info_dict[inst_id]["type"] = pred_type_dict[id]["type"]
                        inst_info_dict[inst_id]["type_prob"] = pred_type_dict[id]["type_prob"]
        elif type_map_path:
            pred_type = np.load(f"{type_map_path}/{image_names[i]}.npy", allow_pickle=True).item()[group]
            for inst_id in list(inst_info_dict.keys()):
                rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
                inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
                inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
                inst_map_crop = inst_map_crop == inst_id
                inst_type = inst_type_crop[inst_map_crop]

                type_list, type_pixels = np.unique(inst_type, return_counts=True)
                type_list = list(zip(type_list, type_pixels))
                type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
                inst_type = type_list[0][0]

                if inst_type == 0:  # ! pick the 2nd most dominant if exist
                    if len(type_list) > 1:
                        inst_type = type_list[1][0]
                type_dict = {v[0]: v[1] for v in type_list}
                type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
                inst_info_dict[inst_id]["type"] = int(inst_type)
                inst_info_dict[inst_id]["type_prob"] = float(type_prob)

        if use_dilation:
            pred_inst_relabel, inst_info_dict = dilation_for_instances(pred_inst, inst_info_dict)
            new_inst_maps.append(pred_inst_relabel)
        else:
            new_inst_maps.append(pred_inst)

        geojson_results.append(inst_info_dict)

    return new_inst_maps, geojson_results


def calculate_instances_for_only_image(inst_maps, image_names, uni_result_path=None, type_map_path=None, ):
    geojson_results = []
    for i, pred_inst in enumerate(inst_maps):
        inst_info_dict: dict[int, dict] = {}
        for inst_id in np.unique(pred_inst):
            if inst_id != 0:
                m = (pred_inst == inst_id)

                # bbox
                rmin, rmax, cmin, cmax = get_bounding_box(m)
                bbox = np.array([[rmin, cmin], [rmax, cmax]], dtype=np.int32)

                # centroid (x,y) using moments on cropped mask
                crop = m[rmin:rmax, cmin:cmax].astype(np.uint8)

                M = cv2.moments(crop)
                if abs(M["m00"]) < 1e-8:
                    cx = (cmin + cmax) / 2.0
                    cy = (rmin + rmax) / 2.0
                else:
                    cx = (M["m10"] / M["m00"]) + cmin
                    cy = (M["m01"] / M["m00"]) + rmin
                centroid = np.array([cx, cy], dtype=np.float32)  # (x,y)

                contours, _ = cv2.findContours(crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) == 0:
                    continue

                cnt = max(contours, key=cv2.contourArea).squeeze(1)  # (N,2) in crop coords

                if cnt.ndim != 2 or cnt.shape[0] < 3:
                    continue
                cnt[:, 0] += cmin  # x
                cnt[:, 1] += rmin  # y
                contour = cnt.astype(np.int32)

                inst_info_dict[int(inst_id)] = {
                    "bbox": bbox,
                    "centroid": centroid,  # (x,y)
                    "contour": contour,  # (x,y)
                    "type": 0,
                    "type_prob": 0
                }

        if uni_result_path:
            with open(f"{uni_result_path}/{image_names[i]}.json", "r", encoding="utf-8") as f:
                pred_type_dict = json.load(f)
                for inst_id in list(inst_info_dict.keys()):
                    id = str(inst_id)
                    if id in pred_type_dict.keys():
                        inst_info_dict[inst_id]["type"] = pred_type_dict[id]["type"]
                        inst_info_dict[inst_id]["type_prob"] = pred_type_dict[id]["type_prob"]
        elif type_map_path:
            pred_type = np.load(f"{type_map_path}/{image_names[i]}.npy", allow_pickle=True).item()["pred"]
            for inst_id in list(inst_info_dict.keys()):
                rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
                inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
                inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
                inst_map_crop = inst_map_crop == inst_id
                inst_type = inst_type_crop[inst_map_crop]

                type_list, type_pixels = np.unique(inst_type, return_counts=True)
                type_list = list(zip(type_list, type_pixels))
                type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
                inst_type = type_list[0][0]

                if inst_type == 0:  # ! pick the 2nd most dominant if exist
                    if len(type_list) > 1:
                        inst_type = type_list[1][0]
                type_dict = {v[0]: v[1] for v in type_list}
                type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
                inst_info_dict[inst_id]["type"] = int(inst_type)
                inst_info_dict[inst_id]["type_prob"] = float(type_prob)

        geojson_results.append(inst_info_dict)

    return geojson_results


def rebuild_inst_info_dict(pred_inst_new, old_pred_inst, old_inst_info_dict):
    """
    根据最终 pred_inst_new 重新构建 inst_info_dict
    type/type_prob 从 old_inst_info_dict 中继承：
    对每个新实例，找和它 overlap 最大的旧实例，继承其 type/type_prob
    """
    new_inst_info_dict = {}

    inst_ids = np.unique(pred_inst_new)
    inst_ids = inst_ids[inst_ids != 0]

    for new_id in inst_ids:
        mask = (pred_inst_new == new_id)
        coords = np.argwhere(mask)

        if coords.shape[0] == 0:
            continue

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1

        centroid_yx = coords.mean(axis=0)
        centroid = [float(centroid_yx[1]), float(centroid_yx[0])]  # [x, y]

        contours = find_contours(mask.astype(np.uint8), 0.5)
        if len(contours) > 0:
            contour = max(contours, key=len)
            contour = np.fliplr(contour)  # (row, col) -> (x, y)
            contour = contour.astype(np.int32)
        else:
            contour = np.zeros((0, 2), dtype=np.int32)

        # 找 overlap 最大的旧实例
        overlapped_old_ids = old_pred_inst[mask]
        overlapped_old_ids = overlapped_old_ids[overlapped_old_ids != 0]

        if len(overlapped_old_ids) > 0:
            unique_ids, counts = np.unique(overlapped_old_ids, return_counts=True)
            best_old_id = unique_ids[np.argmax(counts)]

            old_info = old_inst_info_dict.get(int(best_old_id), {})
            inst_type = old_info.get("type", None)
            type_prob = old_info.get("type_prob", None)
        else:
            inst_type = None
            type_prob = None

        new_inst_info_dict[int(new_id)] = {
            "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
            "centroid": centroid,
            "contour": contour,
            "type": inst_type,
            "type_prob": type_prob,
        }

    return new_inst_info_dict


def dilation_for_instances(pred_inst, inst_info_dict):
    TARGET_TYPES = {5}
    ERODE_RADIUS = 1
    selem = disk(ERODE_RADIUS)

    # 保存原始 pred_inst，用于后面继承 type/type_prob
    old_pred_inst = pred_inst.copy()

    # # -------------------------
    # # Step 1: 对非 target 类型做 erosion
    # # -------------------------
    # pred_inst_processed = pred_inst.copy()
    #
    # for inst_id, info in inst_info_dict.items():
    #     if info["type"] in TARGET_TYPES:
    #         continue
    #
    #     inst_mask = (pred_inst == inst_id)
    #     if inst_mask.sum() < 10:
    #         continue
    #
    #     eroded_mask = erosion(inst_mask, selem)
    #     if eroded_mask.sum() == 0:
    #         continue
    #
    #     pred_inst_processed[inst_mask] = 0
    #     pred_inst_processed[eroded_mask] = inst_id
    #
    # pred_inst = pred_inst_processed

    # -------------------------
    # Step 2: 构建只包含 target type 的 mask
    # -------------------------
    merge_mask = np.zeros_like(pred_inst, dtype=np.int32)
    for inst_id, info in inst_info_dict.items():
        if info["type"] in TARGET_TYPES:
            merge_mask[pred_inst == inst_id] = inst_id

    # -------------------------
    # Step 3: 膨胀后找连通区域
    # -------------------------
    dilated = dilation(merge_mask > 0, footprint=disk(1)).astype(np.uint8)
    merged_label = cc_label(dilated)

    groups = {}
    for inst_id in np.unique(merge_mask):
        if inst_id == 0:
            continue

        inst_bin = (merge_mask == inst_id)
        inst_bin_d = dilation(inst_bin, footprint=disk(1))

        gids = merged_label[inst_bin_d]
        gids = gids[gids != 0]
        if gids.size == 0:
            continue

        gid = np.bincount(gids).argmax()
        groups.setdefault(gid, []).append(int(inst_id))

    # -------------------------
    # Step 4: 执行 merge
    # -------------------------
    pred_inst_new = pred_inst.copy()
    next_id = int(pred_inst_new.max()) + 1

    for gid, id_list in groups.items():
        if len(id_list) <= 1:
            continue

        union_mask = np.isin(pred_inst_new, id_list)
        pred_inst_new[union_mask] = next_id
        next_id += 1

    # -------------------------
    # Step 5: relabel，保证 id 连续
    # -------------------------
    ids = np.unique(pred_inst_new)
    ids = ids[ids != 0]

    pred_inst_relabel = np.zeros_like(pred_inst_new, dtype=np.int32)
    mapping = {old_id: new_id for new_id, old_id in enumerate(ids, start=1)}

    for old_id, new_id in mapping.items():
        pred_inst_relabel[pred_inst_new == old_id] = new_id

    # -------------------------
    # Step 6: 重建新的 inst_info_dict
    # -------------------------
    new_inst_info_dict = rebuild_inst_info_dict(
        pred_inst_new=pred_inst_relabel,
        old_pred_inst=old_pred_inst,
        old_inst_info_dict=inst_info_dict
    )

    return pred_inst_relabel, new_inst_info_dict


def plot_results(
        imgs: list,
        predictions: dict,
        ground_truth: dict,
        img_names: list,
        outdir: str,
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    # convert to rgb and crop to selection
    sample_images = imgs

    pred_sample_binary_map = predictions["nuclei_binary_map"]
    pred_sample_instance_maps = predictions["instance_map"]
    pred_sample_type_map = predictions["type_map"]

    gt_sample_binary_map = ground_truth["nuclei_binary_map"]
    gt_sample_instance_map = ground_truth["instance_map"]
    # gt_sample_type_map = ground_truth["type_map"]

    for i in tqdm.tqdm(range(len(img_names)), desc="Saving plot results"):
        save_ext = ".png"
        img_id = img_names[i]
        patient_id = re.search(r"larger_(\d+_\d+)_LHE.*?\(([^)]+)\)\_\(\d+_\d+\)", img_id).group(1)

        os.makedirs(f'{outdir}/{patient_id}', exist_ok=True)

        binary_gt_gray = (gt_sample_binary_map[i] * 255).astype(np.uint8)
        Image.fromarray(binary_gt_gray).save(f'{outdir}/{patient_id}/binary_gt_{img_id}{save_ext}')

        # Pred：pred_sample_binary_map[i] 是概率 (0~1)，可直接映射到 0~255
        binary_pred_gray = (pred_sample_binary_map[i] * 255).astype(np.uint8)
        Image.fromarray(binary_pred_gray).save(f'{outdir}/{patient_id}/binary_pred_{img_id}{save_ext}')

        def _save_rgb(arr01, save_path):
            """arr01: HxWx3, 浮点 [0,1]；保存为 RGB 图像"""
            im = (arr01 * 255).astype(np.uint8)
            Image.fromarray(im).save(save_path)

        def overlay_instances(inst_map, image, bg_label=0, alpha=0.5):
            """
            inst_map: HxW 整数矩阵，每个细胞一个ID，背景=bg_label
            image:   HxWx3 RGB图像 (ndarray)，例如 orig_img
            alpha:   透明度，越小越透明
            """
            # 重新编号，避免标签不连续
            # inst_map, _, _ = relabel_sequential(inst_map)

            # 叠加彩色实例到原图
            overlay = label2rgb(
                inst_map,
                image=image,
                bg_label=bg_label,
                bg_color=None,  # 背景保持原图
                alpha=alpha,
                kind='overlay'  # 在原图上叠加
            )
            return overlay

        def overlay_boundaries(inst_map, image, color=(1, 0, 0), alpha=0.8, thickness=2):
            """
            inst_map: HxW instance map
            image: HxWx3 RGB
            color: 边界颜色 (R,G,B)
            alpha: 边界强度
            """
            overlay = image.copy().astype(np.float32)

            # 统一到 float [0,1]
            if overlay.max() > 1.0:
                overlay = overlay / 255.0

            boundaries = find_boundaries(inst_map, mode="outer")

            if thickness > 1:
                boundaries = binary_dilation(boundaries, iterations=thickness - 1)

            overlay[boundaries] = color

            return overlay.clip(0, 1)

        # instance_gt = overlay_instances(gt_sample_instance_map[i], sample_images[i], alpha=0.5)
        # instance_pred = overlay_instances(pred_sample_instance_maps[i], sample_images[i], alpha=0.5)
        instance_gt = overlay_boundaries(gt_sample_instance_map[i], sample_images[i])
        instance_pred = overlay_boundaries(pred_sample_instance_maps[i], sample_images[i])
        _save_rgb(instance_gt, f'{outdir}/{patient_id}/inst_gt_{img_id}{save_ext}')
        _save_rgb(instance_pred, f'{outdir}/{patient_id}/inst_pred_{img_id}{save_ext}')

        cell_colors = [
            "#FFFFFF",  # 0 background (white)
            "#1f77b4",  # 1 blue
            "#ff7f0e",  # 2 orange
            "#2ca02c",  # 3 green
            "#8c564b",  # 4 brown
            "#d62728",  # 5 red
            "#9467bd",  # 6 purple
            "#e377c2",  # 7 pink
            "#7f7f7f",  # 8 gray
            "#bcbd22",  # 9 olive
            "#17becf",  # 10 cyan
            "#aec7e8",  # 11 light blue
            "#ffbb78",  # 12 light orange
            "#98df8a",  # 13 light green
            "#ff9896",  # 14 light red
            "#c5b0d5",  # 15 light purple
            "#c49c94",  # 16 light brown
        ]

        def _hex_to_rgb01(hex_color: str):
            """'#RRGGBB' -> (r,g,b) in [0,1]"""
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

        def _build_type_palette(cell_hex_colors, type_map):
            """
            生成类型调色板: shape=(num_classes, 3), 值域[0,1]
            index 0 视为背景色（不会用于涂抹，只用于占位）
            """
            palette = []

            for i in range(len(cell_hex_colors)):
                if i in type_map:
                    palette.append(_hex_to_rgb01(cell_hex_colors[i]))

            palette = np.array(palette, dtype=np.float32)
            return palette

        # type_palette = _build_type_palette(cell_colors, gt_sample_type_map[i])
        # type_gt_overlay = label2rgb(
        #     gt_sample_type_map[i],
        #     image=sample_images[i],
        #     colors=type_palette[1:],  # 跳过 index 0 (背景)
        #     bg_color=None,
        #     bg_label=0,
        #     alpha=0.5,
        #     kind='overlay'
        # )
        # _save_rgb(type_gt_overlay, f"{outdir}/{patient_id}/type_gt_{img_id}{save_ext}")

        # Pred
        type_palette = _build_type_palette(cell_colors, pred_sample_type_map[i])
        type_pred_overlay = label2rgb(
            pred_sample_type_map[i],
            image=sample_images[i],
            colors=type_palette[1:],
            bg_color=None,
            bg_label=0,
            alpha=0.5,
            kind='overlay'
        )
        _save_rgb(type_pred_overlay, f"{outdir}/{patient_id}/type_pred_{img_id}{save_ext}")

        def overlay_type_boundaries(inst_map, type_map, image, cell_colors, alpha=0.8, thickness=2, bg_label=0):
            """
            inst_map: HxW，每个细胞一个 instance id，背景=0
            type_map: HxW，每个像素的 type id
            image: HxWx3 RGB
            cell_colors: 例如你现在定义的 ["#FFFFFF", "#1f77b4", ...]
            """
            from skimage.segmentation import find_boundaries
            from scipy.ndimage import binary_dilation
            import numpy as np

            overlay = image.copy().astype(np.float32)
            if overlay.max() > 1.0:
                overlay = overlay / 255.0

            inst_map = inst_map.astype(np.int32)

            def hex_to_rgb01(hex_color):
                hex_color = hex_color.lstrip("#")
                return np.array([int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)

            inst_ids = np.unique(inst_map)
            inst_ids = inst_ids[inst_ids != bg_label]

            for inst_id in inst_ids:
                inst_mask = (inst_map == inst_id)

                # 这个实例内部的主 type
                inst_types = type_map[inst_mask]
                inst_types = inst_types[inst_types != 0]  # 可选：忽略背景 type
                if len(inst_types) == 0:
                    type_id = 0
                else:
                    type_id = np.bincount(inst_types).argmax()

                color = hex_to_rgb01(cell_colors[type_id])

                boundary = find_boundaries(inst_mask.astype(np.uint8), mode="outer")
                if thickness > 1:
                    boundary = binary_dilation(boundary, iterations=thickness - 1)

                overlay[boundary] = alpha * color + (1 - alpha) * overlay[boundary]

            return overlay.clip(0, 1)

        # type_boundary_gt = overlay_type_boundaries(
        #     gt_sample_instance_map[i],
        #     gt_sample_type_map[i],
        #     sample_images[i],
        #     cell_colors,
        #     alpha=0.8,
        #     thickness=2
        # )
        # _save_rgb(type_boundary_gt, f"{outdir}/{patient_id}/type_boundary_gt_{img_id}{save_ext}")

        type_boundary_pred = overlay_type_boundaries(
            pred_sample_instance_maps[i],
            pred_sample_type_map[i],
            sample_images[i],
            cell_colors,
            alpha=0.8,
            thickness=2
        )
        _save_rgb(type_boundary_pred, f"{outdir}/{patient_id}/type_boundary_pred_{img_id}{save_ext}")


def overlay_inst_dict_on_image(
        image,
        inst_dict,
        cell_colors,
        alpha=0.5,
        draw_contour=False,
        contour_thickness=1,
        draw_text=False,
        text_mode="type",  # "type" or "prob"
):
    """
    只在每个 cell 区域内部做颜色叠加，其他区域保持原图不变
    """

    # 转成 uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
    else:
        image_uint8 = image.copy()

    # 如果是灰度图，转成3通道
    if image_uint8.ndim == 2:
        image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)

    result = image_uint8.copy()

    for cell_id, cell_info in inst_dict.items():
        contour = cell_info.get("contour", None)
        cell_type = int(cell_info.get("type", 0))
        centroid = cell_info.get("centroid", None)
        type_prob = cell_info.get("type_prob", None)

        if contour is None:
            continue
        if cell_type <= 0 or cell_type >= len(cell_colors):
            continue

        contour = np.asarray(contour, dtype=np.int32)
        if contour.ndim == 2:
            contour = contour.reshape((-1, 1, 2))

        # 当前类别颜色，RGB 0-255
        rgb = cell_colors[cell_type]

        # 先做当前 instance 的 mask
        mask = np.zeros(image_uint8.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, color=1, thickness=-1)
        mask_bool = mask.astype(bool)

        # 只在 mask 区域内进行 alpha blend
        original_pixels = result[mask_bool].astype(np.float32)
        color_pixels = np.tile(rgb, (original_pixels.shape[0], 1)).astype(np.float32)

        blended_pixels = (1 - alpha) * original_pixels + alpha * color_pixels
        result[mask_bool] = blended_pixels.astype(np.uint8)

        # 可选：只画边界
        if draw_contour:
            cv2.drawContours(
                result,
                [contour],
                -1,
                color=tuple(int(x) for x in rgb.tolist()),
                thickness=contour_thickness
            )

        # 可选：写字
        if draw_text and centroid is not None:
            cx, cy = int(centroid[0]), int(centroid[1])

            if text_mode == "type":
                text = f"{cell_type}"
            elif text_mode == "prob" and type_prob is not None:
                if isinstance(type_prob, (list, tuple, np.ndarray)):
                    text = f"{np.max(type_prob):.2f}"
                else:
                    text = f"{float(type_prob):.2f}"
            else:
                text = f"{cell_type}"

            cv2.putText(
                result,
                text,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    return result


# def overlay_inst_dict_on_image(image, inst_dict, cell_colors, alpha=0.5):
#     """
#     直接根据 contour + type 把分类结果 overlay 到原图上
#     """
#     if image.dtype != np.uint8:
#         if image.max() <= 1.0:
#             image_uint8 = (image * 255).astype(np.uint8)
#         else:
#             image_uint8 = image.astype(np.uint8)
#     else:
#         image_uint8 = image.copy()
#
#     overlay = image_uint8.copy()
#
#     for cell_id, cell_info in inst_dict.items():
#         contour = cell_info.get("contour", None)
#         cell_type = int(cell_info.get("type", 0))
#
#         if contour is None:
#             continue
#         if cell_type <= 0 or cell_type >= len(cell_colors):
#             continue
#
#         contour = np.asarray(contour, dtype=np.int32)
#         if contour.ndim == 2:
#             contour = contour.reshape((-1, 1, 2))
#
#         rgb = cell_colors[cell_type]
#
#         # OpenCV 用 BGR
#         bgr = (rgb[2], rgb[1], rgb[0])
#
#         cv2.drawContours(
#             image=overlay,
#             contours=[contour],
#             contourIdx=-1,
#             color=bgr,
#             thickness=-1
#         )
#
#     blended = cv2.addWeighted(overlay, alpha, image_uint8, 1 - alpha, 0)
#
#     return blended


def calculate_step_metric(
        gt: dict,
        predictions: dict,
        image_names: list[str],
        magnification: int = 40,
) -> Tuple[dict]:
    """Calculate the metrics for the validation step

    Args:
        predictions (dict): Processed network output
        gt (dict): Ground truth values
        image_names (list(str)): List with image names

    Returns:
        Tuple[dict]:
            * dict: Dictionary with metrics. Structure not fixed yet
    """
    # preparation and device movement
    instance_maps_pred = predictions["instance_map"].copy()
    instance_maps_gt = gt["instance_map"].copy()

    region_names = [re.search(r"larger_(\d+_\d+)_LHE.*?\(([^)]+)\)\_\(\d+_\d+\)", image_name).group(2)
                    for image_name in image_names]

    # segmentation scores
    binary_dice_scores = []  # binary dice scores per image
    binary_jaccard_scores = []  # binary jaccard scores per image
    pq_scores = []  # pq-scores per image
    dq_scores = []  # dq-scores per image
    sq_scores = []  # sq_scores per image

    # detection scores
    paired_all = []  # unique matched index pair
    unpaired_true_all = []  # the index must exist in `true_inst_type_all` and unique
    unpaired_pred_all = []  # the index must exist in `pred_inst_type_all` and unique

    paired_per_image = []
    unpaired_true_per_image = []
    unpaired_pred_per_image = []

    # for detections scores
    true_idx_offset = 0
    pred_idx_offset = 0

    for i in range(len(region_names)):
        # binary dice score: Score for cell detection per image, without background
        pred_binary_map = predictions["nuclei_binary_map"][i]
        target_binary_map = gt["nuclei_binary_map"][i]

        cell_dice = dice_score(
            pred=pred_binary_map,
            gt=target_binary_map
        )
        binary_dice_scores.append(float(cell_dice))

        # binary aji
        cell_jaccard = binary_jaccard_index(
            pred=pred_binary_map,
            gt=target_binary_map
        )
        binary_jaccard_scores.append(float(cell_jaccard))

        # pq values
        if len(np.unique(instance_maps_gt[i])) == 1:
            dq, sq, pq = np.nan, np.nan, np.nan
        else:
            [dq, sq, pq], _ = get_fast_pq(true=instance_maps_gt[i], pred=instance_maps_pred[i])
        pq_scores.append(pq)
        dq_scores.append(dq)
        sq_scores.append(sq)

        # detection scores
        # build arrays safely (0 cells => (0,2) / (0,))
        gt_inst_dict = gt["instance_types"][i]
        pd_inst_dict = predictions["instance_types"][i]

        if len(gt_inst_dict) == 0:
            true_centroids = np.zeros((0, 2), dtype=np.float32)
        else:
            true_centroids = np.asarray([v["centroid"] for v in gt_inst_dict.values()], dtype=np.float32)

        if len(pd_inst_dict) == 0:
            pred_centroids = np.zeros((0, 2), dtype=np.float32)
        else:
            pred_centroids = np.asarray([v["centroid"] for v in pd_inst_dict.values()], dtype=np.float32)

        pairing_radius = 12 if magnification == 40 else 6
        paired, unpaired_true, unpaired_pred = pair_coordinates(true_centroids, pred_centroids, pairing_radius)

        paired_per_image.append(paired.copy())
        unpaired_true_per_image.append(unpaired_true.copy())
        unpaired_pred_per_image.append(unpaired_pred.copy())

        # offset to global indices
        if paired.shape[0] > 0:
            paired = paired.copy()
            paired[:, 0] += true_idx_offset
            paired[:, 1] += pred_idx_offset
            paired_all.append(paired)

        unpaired_true_all.append(unpaired_true + true_idx_offset)
        unpaired_pred_all.append(unpaired_pred + pred_idx_offset)

        # update running offsets by counts in THIS image
        true_idx_offset += true_centroids.shape[0]
        pred_idx_offset += pred_centroids.shape[0]

    paired_all = np.concatenate(paired_all, axis=0) if len(paired_all) else np.zeros((0, 2), np.int32)
    unpaired_true_all = np.concatenate(unpaired_true_all, axis=0) if len(unpaired_true_all) else np.zeros((0,),
                                                                                                          np.int32)
    unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0) if len(unpaired_pred_all) else np.zeros((0,),
                                                                                                          np.int32)

    batch_metrics = {
        "image_names": image_names,
        "binary_dice_scores": binary_dice_scores,
        "binary_jaccard_scores": binary_jaccard_scores,
        "pq_scores": pq_scores,
        "dq_scores": dq_scores,
        "sq_scores": sq_scores,
        "paired_all": paired_all,
        "unpaired_true_all": unpaired_true_all,
        "unpaired_pred_all": unpaired_pred_all,
        "region_names": region_names,
        "paired_per_image": paired_per_image,
        "unpaired_true_per_image": unpaired_true_per_image,
        "unpaired_pred_per_image": unpaired_pred_per_image,
    }

    return batch_metrics


def calculate_step_metric_with_class(
        gt: dict,
        predictions: dict,
        image_names: list[str],
        magnification: int = 40,
        num_classes: int = 8,
        input_imgs: List = None,
        save_crop_root: str = "./pred_cell_crops",
        crop_size: int = 64,
        save_crop_result: bool = False,
) -> Tuple[dict]:
    """Calculate the metrics for the validation step

    Args:
        predictions (dict): Processed network output
        gt (dict): Ground truth values
        image_names (list(str)): List with image names

    Returns:
        Tuple[dict]:
            * dict: Dictionary with metrics. Structure not fixed yet
    """
    # preparation and device movement
    instance_maps_pred = predictions["instance_map"].copy()
    instance_maps_gt = gt["instance_map"].copy()

    region_names = [re.search(r"larger_(\d+_\d+)_LHE.*?\(([^)]+)\)\_\(\d+_\d+\)", image_name).group(2)
                    for image_name in image_names]

    # segmentation scores
    binary_dice_scores = []  # binary dice scores per image
    binary_jaccard_scores = []  # binary jaccard scores per image
    pq_scores = []  # pq-scores per image
    dq_scores = []  # dq-scores per image
    sq_scores = []  # sq_scores per image

    cell_type_pq_scores = []  # pq-scores per cell type and image
    cell_type_dq_scores = []  # dq-scores per cell type and image
    cell_type_sq_scores = []  # sq-scores per cell type and image

    # detection scores
    paired_all = []  # unique matched index pair
    unpaired_true_all = []  # the index must exist in `true_inst_type_all` and unique
    unpaired_pred_all = []  # the index must exist in `pred_inst_type_all` and unique

    true_inst_type_all = []  # each index is 1 independent data point
    pred_inst_type_all = []  # each index is 1 independent data point

    paired_true_type_list = []
    paired_pred_type_list = []
    unpaired_true_type_list = []
    unpaired_pred_type_list = []

    # for detections scores
    true_idx_offset = 0
    pred_idx_offset = 0

    os.makedirs(save_crop_root, exist_ok=True)
    for i in range(len(region_names)):
        # binary dice score: Score for cell detection per image, without background
        pred_binary_map = predictions["nuclei_binary_map"][i]
        target_binary_map = gt["nuclei_binary_map"][i]

        cell_dice = dice_score(
            pred=pred_binary_map,
            gt=target_binary_map
        )
        binary_dice_scores.append(float(cell_dice))

        # binary aji
        cell_jaccard = binary_jaccard_index(
            pred=pred_binary_map,
            gt=target_binary_map
        )
        binary_jaccard_scores.append(float(cell_jaccard))

        # pq values
        if len(np.unique(instance_maps_gt[i])) == 1:
            dq, sq, pq = np.nan, np.nan, np.nan
        else:
            [dq, sq, pq], _ = get_fast_pq(true=instance_maps_gt[i], pred=instance_maps_pred[i])
        pq_scores.append(pq)
        dq_scores.append(dq)
        sq_scores.append(sq)

        # pq values per class (with class 0 beeing background -> should be skipped in the future)
        nuclei_type_pq = []
        nuclei_type_dq = []
        nuclei_type_sq = []
        for j in range(0, num_classes):
            pred_nuclei_instance_class = remap_label(
                predictions["instance_types_nuclei"][i][j, ...]
            )
            target_nuclei_instance_class = remap_label(
                gt["instance_types_nuclei"][i][j, ...]
            )

            # if ground truth is empty, skip from calculation
            if len(np.unique(target_nuclei_instance_class)) == 1:
                pq_tmp = np.nan
                dq_tmp = np.nan
                sq_tmp = np.nan
            else:
                [dq_tmp, sq_tmp, pq_tmp], _ = get_fast_pq(
                    true=target_nuclei_instance_class,
                    pred=pred_nuclei_instance_class,
                    match_iou=0.5,
                )
            nuclei_type_pq.append(pq_tmp)
            nuclei_type_dq.append(dq_tmp)
            nuclei_type_sq.append(sq_tmp)

        # detection scores
        true_centroids = np.array(
            [v["centroid"] for k, v in gt["instance_types"][i].items()]
        )
        true_instance_type = np.array(
            [v["type"] for k, v in gt["instance_types"][i].items()]
        )
        pred_centroids = np.array(
            [v["centroid"] for k, v in predictions["instance_types"][i].items()]
        )
        pred_instance_type = np.array(
            [v["type"] for k, v in predictions["instance_types"][i].items()]
        )

        if true_centroids.shape[0] == 0:
            true_centroids = np.array([[0, 0]])
            true_instance_type = np.array([0])
        if pred_centroids.shape[0] == 0:
            pred_centroids = np.array([[0, 0]])
            pred_instance_type = np.array([0])

        if magnification == 40:
            pairing_radius = 12
        else:
            pairing_radius = 6

        paired, unpaired_true, unpaired_pred = pair_coordinates(
            true_centroids, pred_centroids, pairing_radius
        )

        # ----- per-image types for tissue-level metrics -----
        # paired: shape (K,2) with local indices into true_instance_type / pred_instance_type
        if paired is not None and paired.shape[0] > 0:
            pt_img = true_instance_type[paired[:, 0]]
            pp_img = pred_instance_type[paired[:, 1]]
        else:
            pt_img = np.array([], dtype=true_instance_type.dtype)
            pp_img = np.array([], dtype=pred_instance_type.dtype)

        # unpaired indices are local indices into true_instance_type / pred_instance_type
        if unpaired_true is not None and unpaired_true.shape[0] > 0:
            ut_img = true_instance_type[unpaired_true]
        else:
            ut_img = np.array([], dtype=true_instance_type.dtype)

        if unpaired_pred is not None and unpaired_pred.shape[0] > 0:
            up_img = pred_instance_type[unpaired_pred]
        else:
            up_img = np.array([], dtype=pred_instance_type.dtype)

        paired_true_type_list.append(pt_img)
        paired_pred_type_list.append(pp_img)
        unpaired_true_type_list.append(ut_img)
        unpaired_pred_type_list.append(up_img)

        if save_crop_result:
            # ---------- save pred-centered crops + csv ----------
            if input_imgs is not None:
                gt_items = list(gt["instance_types"][i].items())
                pred_items = list(predictions["instance_types"][i].items())

                true_ids = [k for k, _ in gt_items]
                pred_ids = [k for k, _ in pred_items]

                img_np = input_imgs[i]
                folder_name = image_names[i]
                save_folder = os.path.join(save_crop_root, folder_name)
                os.makedirs(save_folder, exist_ok=True)

                # build pred_local_idx -> matched gt info
                pred_match_dict = {}
                if paired is not None and paired.shape[0] > 0:
                    for gt_local_idx, pred_local_idx in paired:
                        pred_match_dict[int(pred_local_idx)] = {
                            "matched_gt_type": int(true_instance_type[gt_local_idx]),
                            "matched_gt_id": true_ids[gt_local_idx],
                            "is_paired": 1,
                        }

                csv_rows = []

                for pred_local_idx, (pred_cell_id, pred_info) in enumerate(pred_items):
                    centroid = pred_info["centroid"]

                    # centroid format usually [x, y] or [col, row]
                    center_x = float(centroid[0])
                    center_y = float(centroid[1])

                    patch = crop_center_patch(
                        img=img_np,
                        center_x=center_x,
                        center_y=center_y,
                        crop_size=crop_size,
                    )

                    save_img_path = os.path.join(save_folder, f"{pred_cell_id}.png")
                    Image.fromarray(patch).save(save_img_path)

                    if pred_local_idx in pred_match_dict:
                        matched_gt_type = pred_match_dict[pred_local_idx]["matched_gt_type"]
                        matched_gt_id = pred_match_dict[pred_local_idx]["matched_gt_id"]
                        is_paired = 1
                    else:
                        matched_gt_type = -1
                        matched_gt_id = -1
                        is_paired = 0

                    csv_rows.append({
                        "pred_cell_id": pred_cell_id,
                        "centroid_x": center_x,
                        "centroid_y": center_y,
                        "pred_type": int(pred_info["type"]),
                        "matched_gt_type": matched_gt_type,
                        "matched_gt_id": matched_gt_id,
                        "is_paired": is_paired,
                    })

                csv_path = os.path.join(save_folder, "pred_cells_gt_type.csv")
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "pred_cell_id",
                            "centroid_x",
                            "centroid_y",
                            "pred_type",
                            "matched_gt_type",
                            "matched_gt_id",
                            "is_paired",
                        ],
                    )
                    writer.writeheader()
                    writer.writerows(csv_rows)

        # -----------------------------------------------
        true_idx_offset = (
            true_idx_offset + true_inst_type_all[-1].shape[0] if i != 0 else 0
        )
        pred_idx_offset = (
            pred_idx_offset + pred_inst_type_all[-1].shape[0] if i != 0 else 0
        )
        true_inst_type_all.append(true_instance_type)
        pred_inst_type_all.append(pred_instance_type)

        # increment the pairing index statistic
        if paired.shape[0] != 0:  # ! sanity
            paired[:, 0] += true_idx_offset
            paired[:, 1] += pred_idx_offset
            paired_all.append(paired)

        unpaired_true += true_idx_offset
        unpaired_pred += pred_idx_offset
        unpaired_true_all.append(unpaired_true)
        unpaired_pred_all.append(unpaired_pred)

        cell_type_pq_scores.append(nuclei_type_pq)
        cell_type_dq_scores.append(nuclei_type_dq)
        cell_type_sq_scores.append(nuclei_type_sq)

    paired_all = np.concatenate(paired_all, axis=0)
    unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
    unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
    true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
    pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

    batch_metrics = {
        "image_names": image_names,
        "region_names": region_names,

        "binary_dice_scores": binary_dice_scores,
        "binary_jaccard_scores": binary_jaccard_scores,

        "pq_scores": pq_scores,
        "dq_scores": dq_scores,
        "sq_scores": sq_scores,

        "paired_all": paired_all,
        "unpaired_true_all": unpaired_true_all,
        "unpaired_pred_all": unpaired_pred_all,

        "cell_type_pq_scores": cell_type_pq_scores,
        "cell_type_dq_scores": cell_type_dq_scores,
        "cell_type_sq_scores": cell_type_sq_scores,

        "true_inst_type_all": true_inst_type_all,
        "pred_inst_type_all": pred_inst_type_all,

        "paired_true_type_list": paired_true_type_list,
        "paired_pred_type_list": paired_pred_type_list,
        "unpaired_true_type_list": unpaired_true_type_list,
        "unpaired_pred_type_list": unpaired_pred_type_list,
    }

    return batch_metrics


def save_crop_patches(
        predictions,
        input_imgs: List,
        image_names: List,
        save_crop_root: str = "./pred_cell_crops",
        crop_size: int = 64
):
    for i, img in tqdm.tqdm(enumerate(input_imgs), total=len(input_imgs), desc="Saving crop patches"):
        pred_items = list(predictions["instance_types"][i].items())

        save_folder = os.path.join(save_crop_root, image_names[i])
        os.makedirs(save_folder, exist_ok=True)

        for pred_local_idx, (pred_cell_id, pred_info) in enumerate(pred_items):
            centroid = pred_info["centroid"]

            # centroid format usually [x, y] or [col, row]
            center_x = float(centroid[0])
            center_y = float(centroid[1])

            patch = crop_center_patch(
                img=img,
                center_x=center_x,
                center_y=center_y,
                crop_size=crop_size,
            )

            save_img_path = os.path.join(save_folder, f"{pred_cell_id}.png")
            Image.fromarray(patch).save(save_img_path)


def sanitize_folder_name(name: str) -> str:
    """Make image name safe as folder name."""
    name = os.path.splitext(os.path.basename(name))[0]
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    return name


def ensure_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert image to HWC uint8.
    Supports HWC or CHW, grayscale or RGB.
    """
    img = np.asarray(img)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    elif img.ndim == 3:
        # CHW -> HWC
        if img.shape[0] in [1, 3] and img.shape[-1] not in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        # single channel -> 3 channel
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    # convert to uint8 if needed
    if img.dtype != np.uint8:
        img_min, img_max = img.min(), img.max()
        if img_max <= 1.0:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def crop_center_patch(img: np.ndarray, center_x: float, center_y: float, crop_size: int) -> np.ndarray:
    """
    Crop a square patch centered at (center_x, center_y).
    If boundary exceeded, pad with zeros.
    """
    img = ensure_hwc_uint8(img)
    h, w, c = img.shape

    cx = int(round(center_x))
    cy = int(round(center_y))
    half = crop_size // 2

    x1 = cx - half
    y1 = cy - half
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        img = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    patch = img[y1:y2, x1:x2]
    return patch


def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def get_fast_pq(true, pred, match_iou=0.5):
    """
    `match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique
    (1 prediction instance to 1 GT instance mapping).

    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing.

    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.

    Fast computation requires instance IDs are in contiguous orderding
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
    and `by_size` flag has no effect on the result.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    # if there is no background, fixing by adding it
    if 0 not in pred_id_list:
        pred_id_list = [0] + pred_id_list

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1.0e-6)  # good practice?
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


def pair_coordinates(true_xy: np.ndarray, pred_xy: np.ndarray, radius: float) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    true_xy: (Nt,2) in (y,x)
    pred_xy: (Np,2) in (y,x)
    Return:
      paired: (K,2) indices into (true, pred)
      unpaired_true: (Nt-K,)
      unpaired_pred: (Np-K,)
    """
    true_xy = np.asarray(true_xy, dtype=np.float32)
    pred_xy = np.asarray(pred_xy, dtype=np.float32)
    Nt = true_xy.shape[0]
    Np = pred_xy.shape[0]

    if Nt == 0 and Np == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
    if Nt == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.int32), np.arange(Np, dtype=np.int32)
    if Np == 0:
        return np.zeros((0, 2), dtype=np.int32), np.arange(Nt, dtype=np.int32), np.zeros((0,), dtype=np.int32)

    # distance matrix
    d2 = ((true_xy[:, None, :] - pred_xy[None, :, :]) ** 2).sum(axis=-1)
    dist = np.sqrt(d2)

    # cost: dist, but invalidate > radius
    big = 1e6
    cost = dist.copy()
    cost[dist > radius] = big

    r, c = linear_sum_assignment(cost)

    pairs = []
    used_t = set()
    used_p = set()
    for rr, cc in zip(r, c):
        if cost[rr, cc] >= big:
            continue
        pairs.append([rr, cc])
        used_t.add(rr)
        used_p.add(cc)

    paired = np.asarray(pairs, dtype=np.int32) if len(pairs) > 0 else np.zeros((0, 2), dtype=np.int32)
    unpaired_true = np.asarray([i for i in range(Nt) if i not in used_t], dtype=np.int32)
    unpaired_pred = np.asarray([i for i in range(Np) if i not in used_p], dtype=np.int32)
    return paired, unpaired_true, unpaired_pred


def dice_score(pred, gt, eps=1e-6):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)

    if union == 0:
        return 1.0

    return (2 * intersection + eps) / (union + eps)


def binary_jaccard_index(pred, gt, eps=1e-6):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection

    if union == 0:
        return 1.0

    return (intersection + eps) / (union + eps)


def f1_prec_rec_from_counts(tp, fp, fn, eps=1e-8):
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    return f1, prec, rec


def show_metrics(batch_metrics, run_dir, model_name):
    logger = build_inference_logger(run_dir)
    paired_all_global = []  # unique matched index pair
    unpaired_true_all_global = []  # the index must exist in `true_inst_type_all` and unique
    unpaired_pred_all_global = []  # the index must exist in `pred_inst_type_all` and unique

    # unpack batch_metrics
    image_names = batch_metrics["image_names"]

    # dice scores
    binary_dice_scores = batch_metrics["binary_dice_scores"]
    binary_jaccard_scores = batch_metrics["binary_jaccard_scores"]

    # pq scores
    pq_scores = batch_metrics["pq_scores"]
    dq_scores = batch_metrics["dq_scores"]
    sq_scores = batch_metrics["sq_scores"]

    # increment the pairing index statistic
    binary_dice_scores = np.array(binary_dice_scores)
    binary_jaccard_scores = np.array(binary_jaccard_scores)
    pq_scores = np.array(pq_scores)
    dq_scores = np.array(dq_scores)
    sq_scores = np.array(sq_scores)
    paired_all_global.append(batch_metrics["paired_all"])
    unpaired_true_all_global.append(batch_metrics["unpaired_true_all"])
    unpaired_pred_all_global.append(batch_metrics["unpaired_pred_all"])

    paired_all = np.concatenate(paired_all_global, axis=0) if len(paired_all_global) else np.zeros((0, 2), np.int32)
    unpaired_true_all = np.concatenate(unpaired_true_all_global, axis=0) if len(unpaired_true_all_global) else np.zeros(
        (0,), np.int32)
    unpaired_pred_all = np.concatenate(unpaired_pred_all_global, axis=0) if len(unpaired_pred_all_global) else np.zeros(
        (0,), np.int32)

    f1_d, prec_d, rec_d, tp, fp, fn = detection_scores_from_pairs(
        paired_all, unpaired_true_all, unpaired_pred_all
    )

    dataset_metrics = {
        "Binary-Cell-Dice-Mean": float(np.nanmean(binary_dice_scores)),
        "Binary-Cell-Jacard-Mean": float(np.nanmean(binary_jaccard_scores)),
        "bPQ": float(np.nanmean(pq_scores)),
        "bDQ": float(np.nanmean(dq_scores)),
        "bSQ": float(np.nanmean(sq_scores)),
        "f1_detection": float(f1_d),
        "precision_detection": float(prec_d),
        "recall_detection": float(rec_d),
    }

    tissue_types_inf = [t.lower() for t in batch_metrics['region_names']]

    tissue_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    paired_per_image = batch_metrics["paired_per_image"]
    unpaired_true_per_image = batch_metrics["unpaired_true_per_image"]
    unpaired_pred_per_image = batch_metrics["unpaired_pred_per_image"]

    for i, tissue in enumerate(tissue_types_inf):
        tp = int(paired_per_image[i].shape[0])
        fn = int(unpaired_true_per_image[i].shape[0])
        fp = int(unpaired_pred_per_image[i].shape[0])
        tissue_counts[tissue]["tp"] += tp
        tissue_counts[tissue]["fp"] += fp
        tissue_counts[tissue]["fn"] += fn

    # 2) tissue-level segmentation metrics + detection metrics
    tissue_metrics = {}
    tissues = np.unique(np.asarray(tissue_types_inf))

    for tissue in tissues:
        tissue_ids = np.where(np.asarray(tissue_types_inf) == tissue)[0]

        tissue_metrics[tissue] = {}
        tissue_metrics[tissue]["Dice"] = float(np.nanmean(binary_dice_scores[tissue_ids]))
        tissue_metrics[tissue]["Jaccard"] = float(np.nanmean(binary_jaccard_scores[tissue_ids]))
        tissue_metrics[tissue]["bPQ"] = float(np.nanmean(pq_scores[tissue_ids]))
        tissue_metrics[tissue]["bDQ"] = float(np.nanmean(dq_scores[tissue_ids]))
        tissue_metrics[tissue]["bSQ"] = float(np.nanmean(sq_scores[tissue_ids]))

        tp = tissue_counts[tissue]["tp"]
        fp = tissue_counts[tissue]["fp"]
        fn = tissue_counts[tissue]["fn"]
        f1, prec, rec = f1_prec_rec_from_counts(tp, fp, fn)

        tissue_metrics[tissue]["f1_detection"] = float(f1)
        tissue_metrics[tissue]["precision_detection"] = float(prec)
        tissue_metrics[tissue]["recall_detection"] = float(rec)
        tissue_metrics[tissue]["tp"] = int(tp)
        tissue_metrics[tissue]["fp"] = int(fp)
        tissue_metrics[tissue]["fn"] = int(fn)

    combo_name = "ca1_subiculum+ca2_3"
    combo_set = {"ca1_subiculum", "ca2_3"}

    combo_ids = np.where(np.isin(np.asarray(tissue_types_inf), list(combo_set)))[0]

    tissue_metrics[combo_name] = {
        "Dice": float(np.nanmean(binary_dice_scores[combo_ids])),
        "Jaccard": float(np.nanmean(binary_jaccard_scores[combo_ids])),
        "bPQ": float(np.nanmean(pq_scores[combo_ids])),
        "bDQ": float(np.nanmean(dq_scores[combo_ids])),
        "bSQ": float(np.nanmean(sq_scores[combo_ids])),
    }

    tp = sum(tissue_counts[t]["tp"] for t in combo_set)
    fp = sum(tissue_counts[t]["fp"] for t in combo_set)
    fn = sum(tissue_counts[t]["fn"] for t in combo_set)
    f1, prec, rec = f1_prec_rec_from_counts(tp, fp, fn)

    tissue_metrics[combo_name]["f1_detection"] = float(f1)
    tissue_metrics[combo_name]["precision_detection"] = float(prec)
    tissue_metrics[combo_name]["recall_detection"] = float(rec)
    tissue_metrics[combo_name]["tp"] = int(tp)
    tissue_metrics[combo_name]["fp"] = int(fp)
    tissue_metrics[combo_name]["fn"] = int(fn)

    # print final results
    # binary
    logger.info(f"{20 * '*'} Binary Dataset metrics {20 * '*'}")
    [logger.info(f"{f'{k}:': <25} {v}") for k, v in dataset_metrics.items()]

    # tissue -> the PQ values are bPQ values -> what about mBQ?
    logger.info(f"{20 * '*'} Tissue metrics {20 * '*'}")

    header = f"{'Tissue':<20} {'Dice':<10} {'Jaccard':<10} {'bPQ':<10} {'F1_det':<10}"
    logger.info(header)
    logger.info("-" * len(header))

    for key in tissue_metrics:
        logger.info(
            f"{key:<20} "
            f"{tissue_metrics[key]['Dice']:<10.4f} "
            f"{tissue_metrics[key]['Jaccard']:<10.4f} "
            f"{tissue_metrics[key]['bPQ']:<10.4f} "
            f"{tissue_metrics[key]['f1_detection']:<10.4f}"
        )

    # save all folds
    image_metrics = {}
    for idx, image_name in enumerate(image_names):
        image_metrics[image_name] = {
            "Dice": float(binary_dice_scores[idx]),
            "Jaccard": float(binary_jaccard_scores[idx]),
            "bPQ": float(pq_scores[idx]),
        }
    all_metrics = {
        "dataset": dataset_metrics,
        "tissue_metrics": tissue_metrics,
        "image_metrics": image_metrics,
    }

    # saving
    with open(f"{run_dir}/{model_name}/inference_results.json", "w") as outfile:
        json.dump(all_metrics, outfile, indent=2)


def show_metrics_with_class(batch_metrics, run_dir, model_name):
    logger = build_inference_logger(run_dir)
    paired_all_global = []  # unique matched index pair
    unpaired_true_all_global = []  # the index must exist in `true_inst_type_all` and unique
    unpaired_pred_all_global = []  # the index must exist in `pred_inst_type_all` and unique

    true_inst_type_all_global = []  # each index is 1 independent data point
    pred_inst_type_all_global = []  # each index is 1 independent data point

    true_idx_offset = 0
    pred_idx_offset = 0

    tissue_store = defaultdict(lambda: {
        "paired_true": [],
        "paired_pred": [],
        "unpaired_true": [],
        "unpaired_pred": [],
    })

    # unpack batch_metrics
    image_names = batch_metrics["image_names"]

    # dice scores
    binary_dice_scores = batch_metrics["binary_dice_scores"]
    binary_jaccard_scores = batch_metrics["binary_jaccard_scores"]

    # pq scores
    pq_scores = batch_metrics["pq_scores"]
    dq_scores = batch_metrics["dq_scores"]
    sq_scores = batch_metrics["sq_scores"]

    cell_type_pq_scores = batch_metrics["cell_type_pq_scores"]
    cell_type_dq_scores = batch_metrics["cell_type_dq_scores"]
    cell_type_sq_scores = batch_metrics["cell_type_sq_scores"]

    true_inst_type_all_global.append(batch_metrics["true_inst_type_all"])
    pred_inst_type_all_global.append(batch_metrics["pred_inst_type_all"])
    # increment the pairing index statistic
    batch_metrics["paired_all"][:, 0] += true_idx_offset
    batch_metrics["paired_all"][:, 1] += pred_idx_offset
    paired_all_global.append(batch_metrics["paired_all"])

    batch_metrics["unpaired_true_all"] += true_idx_offset
    batch_metrics["unpaired_pred_all"] += pred_idx_offset
    unpaired_true_all_global.append(batch_metrics["unpaired_true_all"])
    unpaired_pred_all_global.append(batch_metrics["unpaired_pred_all"])

    IGNORE_CLASS = 1  # 和你后面一致

    # 这些是 list，长度=当前 batch 的图片数
    tissues_b = [t.lower() for t in batch_metrics['region_names']]
    pt_list = batch_metrics["paired_true_type_list"]
    pp_list = batch_metrics["paired_pred_type_list"]
    ut_list = batch_metrics["unpaired_true_type_list"]
    up_list = batch_metrics["unpaired_pred_type_list"]

    for tissue, pt, pp, ut, up in zip(tissues_b, pt_list, pp_list, ut_list, up_list):
        pt = np.asarray(pt).ravel()
        pp = np.asarray(pp).ravel()
        ut = np.asarray(ut).ravel()
        up = np.asarray(up).ravel()

        # 和你全局一样：pair 要求 true/pred 都不是 ignore
        if pt.size > 0:
            keep_pair = (pt != IGNORE_CLASS) & (pp != IGNORE_CLASS)
            pt = pt[keep_pair]
            pp = pp[keep_pair]

        # unpaired：分别过滤（你全局也是这么写的）
        if ut.size > 0:
            ut = ut[ut != IGNORE_CLASS]
        if up.size > 0:
            up = up[up != IGNORE_CLASS]

        tissue_store[tissue]["paired_true"].append(pt)
        tissue_store[tissue]["paired_pred"].append(pp)
        tissue_store[tissue]["unpaired_true"].append(ut)
        tissue_store[tissue]["unpaired_pred"].append(up)

    paired_all = np.concatenate(paired_all_global, axis=0)
    unpaired_true_all = np.concatenate(unpaired_true_all_global, axis=0)
    unpaired_pred_all = np.concatenate(unpaired_pred_all_global, axis=0)
    true_inst_type_all = np.concatenate(true_inst_type_all_global, axis=0)
    pred_inst_type_all = np.concatenate(pred_inst_type_all_global, axis=0)
    paired_true_type = true_inst_type_all[paired_all[:, 0]]
    paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
    unpaired_true_type = true_inst_type_all[unpaired_true_all]
    unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

    # Start
    IGNORE_CLASS = 1

    def _drop_index_from_vec(vec, drop_idx):
        if vec.ndim != 1:
            vec = np.asarray(vec).ravel()
        if drop_idx < 0 or drop_idx >= vec.shape[0]:
            return vec
        return np.concatenate([vec[:drop_idx], vec[drop_idx + 1:]], axis=0)

    def _drop_index_from_list_of_vecs(vec_list, drop_idx):
        return [_drop_index_from_vec(np.asarray(v), drop_idx) for v in vec_list]

    # 1. Filter instance pairing related
    pair_keep_mask = (paired_true_type != IGNORE_CLASS) & (paired_pred_type != IGNORE_CLASS)
    paired_all = paired_all[pair_keep_mask]
    paired_true_type = paired_true_type[pair_keep_mask]
    paired_pred_type = paired_pred_type[pair_keep_mask]

    unpaired_true_mask = true_inst_type_all[unpaired_true_all] != IGNORE_CLASS
    unpaired_pred_mask = pred_inst_type_all[unpaired_pred_all] != IGNORE_CLASS
    unpaired_true_all = unpaired_true_all[unpaired_true_mask]
    unpaired_pred_all = unpaired_pred_all[unpaired_pred_mask]
    unpaired_true_type = true_inst_type_all[unpaired_true_all]
    unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

    # 2. Delete the IGNORE class of the PQ/DQ/SQ vector
    cell_type_pq_scores = _drop_index_from_list_of_vecs(cell_type_pq_scores, IGNORE_CLASS)
    cell_type_dq_scores = _drop_index_from_list_of_vecs(cell_type_dq_scores, IGNORE_CLASS)
    cell_type_sq_scores = _drop_index_from_list_of_vecs(cell_type_sq_scores, IGNORE_CLASS)
    # Finish

    # increment the pairing index statistic
    binary_dice_scores = np.array(binary_dice_scores)
    binary_jaccard_scores = np.array(binary_jaccard_scores)
    pq_scores = np.array(pq_scores)
    dq_scores = np.array(dq_scores)
    sq_scores = np.array(sq_scores)

    f1_d, prec_d, rec_d = cell_detection_scores(
        paired_true=paired_true_type,
        paired_pred=paired_pred_type,
        unpaired_true=unpaired_true_type,
        unpaired_pred=unpaired_pred_type,
    )

    y_true_cls = np.asarray(paired_true_type)
    y_pred_cls = np.asarray(paired_pred_type)

    # Macro: 每类等权
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_cls, y_pred_cls, labels=np.arange(2, 7), average="macro", zero_division=0
    )

    # Weighted: 按support加权（=你说的“根据数量加权平均”）
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true_cls, y_pred_cls, labels=np.arange(2, 7), average="weighted", zero_division=0
    )

    dataset_metrics = {
        "Binary-Cell-Dice-Mean": float(np.nanmean(binary_dice_scores)),
        "Binary-Cell-Jacard-Mean": float(np.nanmean(binary_jaccard_scores)),
        "bPQ": float(np.nanmean(pq_scores)),
        "bDQ": float(np.nanmean(dq_scores)),
        "bSQ": float(np.nanmean(sq_scores)),

        "mPQ": float(np.nanmean([np.nanmean(pq) for pq in cell_type_pq_scores])),
        "mDQ": float(np.nanmean([np.nanmean(dq) for dq in cell_type_dq_scores])),
        "mSQ": float(np.nanmean([np.nanmean(sq) for sq in cell_type_sq_scores])),

        "f1_detection": float(f1_d),
        "precision_detection": float(prec_d),
        "recall_detection": float(rec_d),

        "f1_classification_macro": float(f1_macro),
        "precision_classification_macro": float(prec_macro),
        "recall_classification_macro": float(rec_macro),
        "f1_classification_weighted": float(f1_w),
        "precision_classification_weighted": float(prec_w),
        "recall_classification_weighted": float(rec_w),
    }

    def _safe_concat(xs):
        xs = [np.asarray(x).ravel() for x in xs if np.asarray(x).size > 0]
        return np.concatenate(xs, axis=0) if len(xs) > 0 else np.array([], dtype=np.int64)

    # calculate tissue metrics
    tissue_types_inf = [t.lower() for t in batch_metrics['region_names']]
    tissue_types = np.unique(np.asarray(tissue_types_inf))
    tissue_metrics = {}
    for tissue in tissue_types:
        tissue = tissue.lower()
        tissue_ids = np.where(np.asarray(tissue_types_inf) == tissue)

        tissue_metrics[tissue] = {}
        tissue_metrics[tissue]["Dice"] = float(np.nanmean(binary_dice_scores[tissue_ids]))
        tissue_metrics[tissue]["Jaccard"] = float(np.nanmean(binary_jaccard_scores[tissue_ids]))
        tissue_metrics[tissue]["mPQ"] = float(
            np.nanmean([np.nanmean(pq) for pq in np.array(cell_type_pq_scores)[tissue_ids]])
        )
        tissue_metrics[tissue]["bPQ"] = float(np.nanmean(pq_scores[tissue_ids]))

        # -------- NEW: tissue-level detection/classification ----------
        pt = _safe_concat(tissue_store[tissue]["paired_true"])
        pp = _safe_concat(tissue_store[tissue]["paired_pred"])
        ut = _safe_concat(tissue_store[tissue]["unpaired_true"])
        up = _safe_concat(tissue_store[tissue]["unpaired_pred"])

        f1_d_t, prec_d_t, rec_d_t = cell_detection_scores(
            paired_true=pt, paired_pred=pp, unpaired_true=ut, unpaired_pred=up
        )
        tissue_metrics[tissue]["f1_detection"] = float(f1_d_t)
        tissue_metrics[tissue]["precision_detection"] = float(prec_d_t)
        tissue_metrics[tissue]["recall_detection"] = float(rec_d_t)

        # classification（paired only）
        if pt.size == 0:
            tissue_metrics[tissue]["f1_classification_macro"] = float("nan")
            tissue_metrics[tissue]["precision_classification_macro"] = float("nan")
            tissue_metrics[tissue]["recall_classification_macro"] = float("nan")
            tissue_metrics[tissue]["f1_classification_weighted"] = float("nan")
            tissue_metrics[tissue]["precision_classification_weighted"] = float("nan")
            tissue_metrics[tissue]["recall_classification_weighted"] = float("nan")
        else:
            prec_macro_t, rec_macro_t, f1_macro_t, _ = precision_recall_fscore_support(
                pt, pp, average="macro", zero_division=0
            )
            prec_w_t, rec_w_t, f1_w_t, _ = precision_recall_fscore_support(
                pt, pp, average="weighted", zero_division=0
            )
            tissue_metrics[tissue]["f1_classification_macro"] = float(f1_macro_t)
            tissue_metrics[tissue]["precision_classification_macro"] = float(prec_macro_t)
            tissue_metrics[tissue]["recall_classification_macro"] = float(rec_macro_t)
            tissue_metrics[tissue]["f1_classification_weighted"] = float(f1_w_t)
            tissue_metrics[tissue]["precision_classification_weighted"] = float(prec_w_t)
            tissue_metrics[tissue]["recall_classification_weighted"] = float(rec_w_t)

    combo_name = "ca1_subiculum+ca2_3"
    combo_set = {"ca1_subiculum", "ca2_3"}

    combo_ids = np.where(np.isin(np.asarray(tissue_types_inf), list(combo_set)))[0]

    tissue_metrics[combo_name] = {}

    # Dice/Jaccard/bPQ 直接按图取子集
    tissue_metrics[combo_name]["Dice"] = float(np.nanmean(binary_dice_scores[combo_ids]))
    tissue_metrics[combo_name]["Jaccard"] = float(np.nanmean(binary_jaccard_scores[combo_ids]))
    tissue_metrics[combo_name]["bPQ"] = float(np.nanmean(pq_scores[combo_ids]))
    tissue_metrics[combo_name]["bDQ"] = float(np.nanmean(dq_scores[combo_ids]))
    tissue_metrics[combo_name]["bSQ"] = float(np.nanmean(sq_scores[combo_ids]))

    # mPQ/mDQ/mSQ：对每图的“每类PQ向量”先取均值，再对图取均值（与你 dataset-level 的写法一致）
    cell_type_pq_arr = np.array(cell_type_pq_scores, dtype=object)
    cell_type_dq_arr = np.array(cell_type_dq_scores, dtype=object)
    cell_type_sq_arr = np.array(cell_type_sq_scores, dtype=object)

    tissue_metrics[combo_name]["mPQ"] = float(np.nanmean([np.nanmean(v) for v in cell_type_pq_arr[combo_ids]]))
    tissue_metrics[combo_name]["mDQ"] = float(np.nanmean([np.nanmean(v) for v in cell_type_dq_arr[combo_ids]]))
    tissue_metrics[combo_name]["mSQ"] = float(np.nanmean([np.nanmean(v) for v in cell_type_sq_arr[combo_ids]]))

    def _safe_concat(xs):
        xs = [np.asarray(x).ravel() for x in xs if np.asarray(x).size > 0]
        return np.concatenate(xs, axis=0) if len(xs) > 0 else np.array([], dtype=np.int64)

    pt = _safe_concat(tissue_store["ca1"]["paired_true"] + tissue_store["ca2_3"]["paired_true"])
    pp = _safe_concat(tissue_store["ca1"]["paired_pred"] + tissue_store["ca2_3"]["paired_pred"])
    ut = _safe_concat(tissue_store["ca1"]["unpaired_true"] + tissue_store["ca2_3"]["unpaired_true"])
    up = _safe_concat(tissue_store["ca1"]["unpaired_pred"] + tissue_store["ca2_3"]["unpaired_pred"])

    f1_d_t, prec_d_t, rec_d_t = cell_detection_scores(
        paired_true=pt, paired_pred=pp, unpaired_true=ut, unpaired_pred=up
    )
    tissue_metrics[combo_name]["f1_detection"] = float(f1_d_t)
    tissue_metrics[combo_name]["precision_detection"] = float(prec_d_t)
    tissue_metrics[combo_name]["recall_detection"] = float(rec_d_t)

    if pt.size == 0:
        tissue_metrics[combo_name]["f1_classification_macro"] = float("nan")
        tissue_metrics[combo_name]["precision_classification_macro"] = float("nan")
        tissue_metrics[combo_name]["recall_classification_macro"] = float("nan")
        tissue_metrics[combo_name]["f1_classification_weighted"] = float("nan")
        tissue_metrics[combo_name]["precision_classification_weighted"] = float("nan")
        tissue_metrics[combo_name]["recall_classification_weighted"] = float("nan")
    else:
        prec_macro_t, rec_macro_t, f1_macro_t, _ = precision_recall_fscore_support(
            pt, pp, average="macro", zero_division=0
        )
        prec_w_t, rec_w_t, f1_w_t, _ = precision_recall_fscore_support(
            pt, pp, average="weighted", zero_division=0
        )
        tissue_metrics[combo_name]["f1_classification_macro"] = float(f1_macro_t)
        tissue_metrics[combo_name]["precision_classification_macro"] = float(prec_macro_t)
        tissue_metrics[combo_name]["recall_classification_macro"] = float(rec_macro_t)
        tissue_metrics[combo_name]["f1_classification_weighted"] = float(f1_w_t)
        tissue_metrics[combo_name]["precision_classification_weighted"] = float(prec_w_t)
        tissue_metrics[combo_name]["recall_classification_weighted"] = float(rec_w_t)

    # calculate nuclei metrics
    nuclei_types = {
        'background': 0,
        '??': 1,
        'neuron': 2,
        'oligo': 3,
        '?? (debris)': 4,
        'bloodvesselregion': 5,
        'fimbria boundary cells': 6,
        'Ignore*': 7,
    }
    nuclei_metrics_d = {}
    nuclei_metrics_pq = {}
    nuclei_metrics_dq = {}
    nuclei_metrics_sq = {}
    for nuc_name, nuc_type in nuclei_types.items():
        if nuc_name.lower() == "background":
            continue

        # nuclei_metrics_pq[nuc_name] = np.nanmean(
        #     [pq[nuc_type] for pq in cell_type_pq_scores]
        # )
        # nuclei_metrics_dq[nuc_name] = np.nanmean(
        #     [dq[nuc_type] for dq in cell_type_dq_scores]
        # )
        # nuclei_metrics_sq[nuc_name] = np.nanmean(
        #     [sq[nuc_type] for sq in cell_type_sq_scores]
        # )

        if nuc_type == IGNORE_CLASS:
            nuclei_metrics_pq[nuc_name] = 0
            nuclei_metrics_dq[nuc_name] = 0
            nuclei_metrics_sq[nuc_name] = 0
            nuclei_metrics_d[nuc_name] = {
                "f1_cell": 0,
                "prec_cell": 0,
                "rec_cell": 0,
            }
            continue

        nuclei_metrics_pq[nuc_name] = np.nanmean(
            [pq[nuc_type - 1] for pq in cell_type_pq_scores]
        )
        nuclei_metrics_dq[nuc_name] = np.nanmean(
            [dq[nuc_type - 1] for dq in cell_type_dq_scores]
        )
        nuclei_metrics_sq[nuc_name] = np.nanmean(
            [sq[nuc_type - 1] for sq in cell_type_sq_scores]
        )

        f1_cell, prec_cell, rec_cell = cell_type_detection_scores(
            paired_true_type,
            paired_pred_type,
            unpaired_true_type,
            unpaired_pred_type,
            nuc_type,
        )
        nuclei_metrics_d[nuc_name] = {
            "f1_cell": f1_cell,
            "prec_cell": prec_cell,
            "rec_cell": rec_cell,
        }

    # print final results
    # binary
    logger.info(f"{20 * '*'} Binary Dataset metrics {20 * '*'}")
    [logger.info(f"{f'{k}:': <25} {v}") for k, v in dataset_metrics.items()]

    # tissue -> the PQ values are bPQ values -> what about mBQ?
    logger.info(f"{20 * '*'} Tissue metrics {20 * '*'}")

    header = f"{'Tissue':<20} {'Dice':<10} {'Jaccard':<10} {'bPQ':<10} {'mPQ':<10} {'F1_det':<10} {'F1_cla':<10}"
    logger.info(header)
    logger.info("-" * len(header))

    for key in tissue_metrics:
        logger.info(
            f"{key:<20} "
            f"{tissue_metrics[key]['Dice']:<10.4f} "
            f"{tissue_metrics[key]['Jaccard']:<10.4f} "
            f"{tissue_metrics[key]['bPQ']:<10.4f} "
            f"{tissue_metrics[key]['mPQ']:<10.4f}"
            f"{tissue_metrics[key]['f1_detection']:<10.4f}"
            f"{tissue_metrics[key]['f1_classification_weighted']:<10.4f}"
        )

    # save all folds
    image_metrics = {}
    for idx, image_name in enumerate(image_names):
        image_metrics[image_name] = {
            "Dice": float(binary_dice_scores[idx]),
            "Jaccard": float(binary_jaccard_scores[idx]),
            "bPQ": float(pq_scores[idx]),
            "mPQ": float(np.nanmean(cell_type_pq_scores[idx])),
        }
    all_metrics = {
        "dataset": dataset_metrics,
        "tissue_metrics": tissue_metrics,
        "image_metrics": image_metrics,
        "nuclei_metrics_pq": nuclei_metrics_pq,
        "nuclei_metrics_d": nuclei_metrics_d,
    }

    # saving
    with open(f"{run_dir}/{model_name}/inference_results.json", "w") as outfile:
        json.dump(all_metrics, outfile, indent=2)


def cell_detection_scores(
        paired_true, paired_pred, unpaired_true, unpaired_pred, w: List = [1, 1]
):
    tp_d = paired_pred.shape[0]
    fp_d = unpaired_pred.shape[0]
    fn_d = unpaired_true.shape[0]

    # tp_tn_dt = (paired_pred == paired_true).sum()
    # fp_fn_dt = (paired_pred != paired_true).sum()
    prec_d = tp_d / (tp_d + fp_d)
    rec_d = tp_d / (tp_d + fn_d)

    f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

    return f1_d, prec_d, rec_d


def cell_type_detection_scores(
        paired_true,
        paired_pred,
        unpaired_true,
        unpaired_pred,
        type_id,
        w: List = [2, 2, 1, 1],
        exhaustive: bool = True,
):
    type_samples = (paired_true == type_id) | (paired_pred == type_id)

    paired_true = paired_true[type_samples]
    paired_pred = paired_pred[type_samples]

    tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
    tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
    fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
    fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

    if not exhaustive:
        ignore = (paired_true == -1).sum()
        fp_dt -= ignore

    fp_d = (unpaired_pred == type_id).sum()  #
    fn_d = (unpaired_true == type_id).sum()

    prec_type = (tp_dt + tn_dt) / (tp_dt + tn_dt + w[0] * fp_dt + w[2] * fp_d)
    rec_type = (tp_dt + tn_dt) / (tp_dt + tn_dt + w[1] * fn_dt + w[3] * fn_d)

    f1_type = (2 * (tp_dt + tn_dt)) / (
            2 * (tp_dt + tn_dt) + w[0] * fp_dt + w[1] * fn_dt + w[2] * fp_d + w[3] * fn_d
    )
    return f1_type, prec_type, rec_type


def detection_scores_from_pairs(paired_all, unpaired_true_all, unpaired_pred_all, eps=1e-8):
    tp = int(paired_all.shape[0])
    fn = int(unpaired_true_all.shape[0])
    fp = int(unpaired_pred_all.shape[0])

    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    return f1, prec, rec, tp, fp, fn


def build_inference_logger(run_dir):
    logger = logging.getLogger("cellpose_inference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")

    # terminal handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    # file handler
    fh = logging.FileHandler(Path(run_dir) / "inference.log")
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def generate_geojson(predictions, base_output_path, image_names):
    cell_colors = {
        0: {
            'name': 'Background',
            'color': [255, 255, 255]
        },
        1: {
            'name': '??',
            'color': [31, 119, 180]
        },
        2: {
            'name': 'Neuron',
            'color': [255, 127, 14]
        },
        3: {
            'name': 'Oligo',
            'color': [44, 160, 44]
        },
        4: {
            'name': '?? (debris)',
            'color': [140, 86, 75]
        },
        5: {
            'name': 'Blood Vessel',
            'color': [214, 39, 40]
        },
        6: {
            'name': 'Fimbria Boundary',
            'color': [148, 103, 189]
        },
        7: {
            'name': 'Ignore*',
            'color': [227, 119, 194]
        }
    }
    for idx, batch_instance in tqdm.tqdm(enumerate(predictions['instance_types']), desc="Saving geojson results"):
        features = []

        m = re.search(r"larger_(\d+_\d+)_LHE.*?\(([^)]+)\)_\(([^)]+)\)\_\(\d+_\d+\)", image_names[idx])
        patient_id = m.group(1)
        patient_type = m.group(2)
        region_name = m.group(3)

        os.makedirs(os.path.join(base_output_path, patient_type, patient_id, region_name), exist_ok=True)

        for key in batch_instance.keys():
            # 提取轮廓坐标
            instance = batch_instance[key]
            contour_points = instance['contour']

            # 确保轮廓是封闭的
            if not np.array_equal(contour_points[0], contour_points[-1]):
                closed_contour = np.concatenate([contour_points, [contour_points[0]]], axis=0)
            else:
                closed_contour = contour_points

            # 转换坐标格式
            coordinates = [[float(point[0]), float(point[1])] for point in closed_contour]

            # 创建特征
            feature = {
                "type": "Feature",
                "id": str(key),
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates]
                },
                "properties": {
                    "name": str(key),
                    "objectType": "annotation",
                    "classification": cell_colors[instance['type']],
                    "type_prob": float(instance['type_prob']),
                    "centroid": [float(instance['centroid'][0]), float(instance['centroid'][1])]
                },

            }

            features.append(feature)

        # 创建GeoJSON对象
        geojson_obj = {
            "type": "FeatureCollection",
            "features": features
        }

        # 保存为文件
        with open(
                f"{base_output_path}/{patient_type}/{patient_id}/{region_name}/{image_names[idx].split('.png')[0]}).geojson",
                'w') as f:
            json.dump(geojson_obj, f, indent=2)
