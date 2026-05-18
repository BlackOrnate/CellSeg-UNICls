from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.utils.io import ensure_dir, load_array


def _crop_with_padding(image: np.ndarray, center_x: float, center_y: float, crop_size: int, pad_value: int = 0) -> np.ndarray:
    half = crop_size // 2
    h, w = image.shape[:2]
    x1, x2 = int(round(center_x)) - half, int(round(center_x)) - half + crop_size
    y1, y2 = int(round(center_y)) - half, int(round(center_y)) - half + crop_size

    if image.ndim == 2:
        out = np.full((crop_size, crop_size), pad_value, dtype=image.dtype)
    else:
        out = np.full((crop_size, crop_size, image.shape[2]), pad_value, dtype=image.dtype)

    src_x1, src_x2 = max(0, x1), min(w, x2)
    src_y1, src_y2 = max(0, y1), min(h, y2)
    dst_x1, dst_y1 = src_x1 - x1, src_y1 - y1
    out[dst_y1:dst_y1 + (src_y2 - src_y1), dst_x1:dst_x1 + (src_x2 - src_x1)] = image[src_y1:src_y2, src_x1:src_x2]
    return out


def instance_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(ys.mean())


def extract_instance_crops(
    image_dir: str | Path,
    instance_dir: str | Path,
    output_dir: str | Path,
    crop_size: int = 64,
    split: str = "test",
    labels_csv: str | Path | None = None,
) -> None:
    image_dir = Path(image_dir)
    instance_dir = Path(instance_dir)
    output_dir = Path(output_dir) / str(crop_size) / split
    label_df = pd.read_csv(labels_csv) if labels_csv else None

    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])
    for image_path in tqdm(image_paths, desc="Extracting instance crops"):
        stem = image_path.stem
        inst_path = instance_dir / f"{stem}.npy"
        if not inst_path.exists():
            continue
        image = np.asarray(Image.open(image_path).convert("RGB"))
        inst_map = load_array(inst_path).astype(np.int32)
        patch_out = ensure_dir(output_dir / stem)

        rows = []
        for inst_id in np.unique(inst_map):
            if inst_id == 0:
                continue
            cx, cy = instance_centroid(inst_map == inst_id)
            crop = _crop_with_padding(image, cx, cy, crop_size)
            Image.fromarray(crop).save(patch_out / f"{int(inst_id)}.png")
            matched_gt_type = -1
            if label_df is not None and {"patch_name", "pred_cell_id", "matched_gt_type"}.issubset(label_df.columns):
                matched = label_df[(label_df["patch_name"] == stem) & (label_df["pred_cell_id"] == int(inst_id))]
                if len(matched) > 0:
                    matched_gt_type = int(matched.iloc[0]["matched_gt_type"])
            rows.append({"pred_cell_id": int(inst_id), "matched_gt_type": matched_gt_type})
        pd.DataFrame(rows).to_csv(patch_out / "pred_cells_gt_type.csv", index=False)
