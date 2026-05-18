from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import build_image_transform


class UNICropDataset(Dataset):
    """Dataset for instance-centered RGB crops before UNI2 embedding extraction.

    Expected crop layout follows the original project:

    crop_root/{patch_size}/{split}/{patch_name}/{cell_id}.png
    crop_root/{patch_size}/{split}/{patch_name}/pred_cells_gt_type.csv

    The CSV should include at least `pred_cell_id` and `matched_gt_type` columns.
    """

    def __init__(
        self,
        crop_root: str | Path,
        split: str,
        patch_size: int = 64,
        mode: Literal["train", "test"] = "train",
        resize_to: int | None = 224,
        ignore_label: int = -1,
    ) -> None:
        self.crop_root = Path(crop_root)
        self.split = split
        self.patch_size = patch_size
        self.mode = mode
        self.ignore_label = ignore_label
        self.transform = build_image_transform(resize_to)

        self.image_paths: list[Path] = []
        self.cell_ids: list[int] = []
        self.labels: list[int] = []
        self.patch_names: list[str] = []

        split_root = self.crop_root / str(patch_size) / split
        if not split_root.exists():
            raise FileNotFoundError(f"Crop split directory not found: {split_root}")

        for patch_dir in sorted(split_root.iterdir()):
            if not patch_dir.is_dir():
                continue
            csv_path = patch_dir / "pred_cells_gt_type.csv"
            label_df = pd.read_csv(csv_path) if csv_path.exists() else None
            for image_path in sorted(patch_dir.glob("*.png")):
                cell_id = int(image_path.stem)
                label = self.ignore_label
                if label_df is not None and "pred_cell_id" in label_df and "matched_gt_type" in label_df:
                    matched = label_df.loc[label_df["pred_cell_id"] == cell_id, "matched_gt_type"]
                    if len(matched) > 0:
                        label = int(matched.iloc[0])
                if mode == "train" and label == self.ignore_label:
                    continue
                self.image_paths.append(image_path)
                self.cell_ids.append(cell_id)
                self.labels.append(label)
                self.patch_names.append(patch_dir.name)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return (
            self.transform(image),
            int(self.labels[idx]),
            self.patch_names[idx],
            int(self.cell_ids[idx]),
        )


class UNIEmbeddingDataset(Dataset):
    """Dataset for saved UNI2 embeddings.

    Expected layout:
    embedding_root/uni_embedding_results_train/{split}/{patch_name}/{cell_id}.pt
    embedding_root/uni_embedding_results_test/{split}/{patch_name}/{cell_id}.pt
    """

    def __init__(self, embedding_root: str | Path, split: str, mode: Literal["train", "test"] = "train") -> None:
        self.embedding_root = Path(embedding_root)
        self.split = split
        self.mode = mode
        folder_name = "uni_embedding_results_train" if mode == "train" else "uni_embedding_results_test"
        split_root = self.embedding_root / folder_name / split
        if not split_root.exists():
            raise FileNotFoundError(f"Embedding directory not found: {split_root}")
        self.feature_paths = sorted(split_root.glob("*/*.pt"))

    def __len__(self) -> int:
        return len(self.feature_paths)

    def __getitem__(self, idx: int):
        data = torch.load(self.feature_paths[idx], map_location="cpu")
        return (
            data["feat"].float(),
            int(data["label"]),
            str(data["patch_name"] if "patch_name" in data else data.get("path_name", "")),
            int(data["cell_id"]),
        )


def collate_with_metadata(batch):
    images_or_feats, labels, patch_names, cell_ids = zip(*batch)
    return torch.stack(images_or_feats, dim=0), torch.tensor(labels, dtype=torch.long), list(patch_names), list(cell_ids)
