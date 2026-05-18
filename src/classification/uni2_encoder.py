from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from src.utils.io import ensure_dir


class UNI2FeatureExtractor(nn.Module):
    """Thin wrapper around a local UNI/UNI2 encoder loader.

    The original project imported `get_encoder` from `models.uni`. Because UNI2 weights are
    usually downloaded or stored locally, they are not bundled here. Put your local UNI loader
    under `models/uni` or adapt `load_uni2_encoder` below.
    """

    def __init__(self, encoder_name: str = "uni2-h", device: str | torch.device = "cuda") -> None:
        super().__init__()
        self.device = torch.device(device)
        self.encoder, self.transform = load_uni2_encoder(encoder_name, self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(images.to(self.device))


def load_uni2_encoder(encoder_name: str = "uni2-h", device: torch.device | str = "cuda"):
    try:
        from models.uni import get_encoder  # type: ignore
        encoder, transform = get_encoder(enc_name=encoder_name, device=device)
        return encoder.to(device), transform
    except Exception as exc:
        raise RuntimeError(
            "UNI2 encoder could not be loaded. Add your local `models/uni` loader or edit "
            "src/classification/uni2_encoder.py to load the UNI2 checkpoint used in your environment."
        ) from exc


def save_uni_features(encoder: nn.Module, dataloader, device: str | torch.device, save_root: str | Path, split: str, mode: str) -> None:
    save_root = Path(save_root)
    folder_name = "uni_embedding_results_train" if mode == "train" else "uni_embedding_results_test"
    encoder.eval()
    with torch.no_grad():
        for images, labels, patch_names, cell_ids in tqdm(dataloader, total=len(dataloader), desc=f"Extracting UNI2 features: {split}"):
            feats = encoder(images.to(device)).detach().cpu()
            for feat, label, patch_name, cell_id in zip(feats, labels, patch_names, cell_ids):
                out_dir = ensure_dir(save_root / folder_name / split / str(patch_name))
                torch.save(
                    {"feat": feat, "label": int(label), "patch_name": str(patch_name), "cell_id": int(cell_id)},
                    out_dir / f"{int(cell_id)}.pt",
                )
