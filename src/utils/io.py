from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=True)
    from PIL import Image
    return np.asarray(Image.open(path))


def save_array(array: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix.lower() == ".npy":
        np.save(path, array)
    else:
        from PIL import Image
        Image.fromarray(array).save(path)
