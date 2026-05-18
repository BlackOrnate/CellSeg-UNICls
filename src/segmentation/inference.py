from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.io import ensure_dir


def infer_cellpose_sam(image_dir: str | Path, checkpoint: str | Path, output_dir: str | Path, batch_size: int = 32) -> None:
    try:
        from cellpose import models
    except Exception as exc:
        raise RuntimeError("Please install Cellpose/Cellpose-SAM before inference.") from exc

    image_dir = Path(image_dir)
    output_dir = ensure_dir(output_dir)
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])
    images = [np.asarray(Image.open(p).convert("RGB")) for p in image_paths]
    model = models.CellposeModel(gpu=True, pretrained_model=str(checkpoint))
    masks, flows, styles = model.eval(images, batch_size=batch_size)
    for path, mask in tqdm(list(zip(image_paths, masks)), desc="Saving Cellpose-SAM instance maps"):
        np.save(output_dir / f"{path.stem}.npy", mask.astype(np.int32))
