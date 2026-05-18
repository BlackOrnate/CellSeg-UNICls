from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.segmentation import find_boundaries

from src.utils.io import ensure_dir

DEFAULT_COLORS = {
    0: (255, 255, 255),
    1: (31, 119, 180),
    2: (255, 127, 14),
    3: (44, 160, 44),
    4: (140, 86, 75),
    5: (214, 39, 40),
    6: (148, 103, 189),
}


def overlay_boundaries(image: np.ndarray, instance_map: np.ndarray, color=(255, 0, 0), thickness: int = 1) -> np.ndarray:
    out = image.copy().astype(np.uint8)
    boundaries = find_boundaries(instance_map, mode="outer")
    if thickness > 1:
        boundaries = binary_dilation(boundaries, iterations=thickness - 1)
    out[boundaries] = np.array(color, dtype=np.uint8)
    return out


def overlay_type_map(image: np.ndarray, type_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = image.copy().astype(np.float32)
    color_layer = np.zeros_like(out)
    mask = type_map > 0
    for type_id, color in DEFAULT_COLORS.items():
        color_layer[type_map == type_id] = color
    out[mask] = (1 - alpha) * out[mask] + alpha * color_layer[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def save_overlay(image_path: str | Path, instance_map: np.ndarray, output_path: str | Path, type_map: np.ndarray | None = None) -> None:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    overlay = overlay_type_map(image, type_map) if type_map is not None else overlay_boundaries(image, instance_map)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    Image.fromarray(overlay).save(output_path)
