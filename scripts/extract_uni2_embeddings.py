from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse

import torch
from torch.utils.data import DataLoader

from src.classification.uni2_encoder import UNI2FeatureExtractor, save_uni_features
from src.data.dataset import UNICropDataset, collate_with_metadata


def main():
    parser = argparse.ArgumentParser(description="Extract frozen UNI2 embeddings from instance-centered crops.")
    parser.add_argument("--crop_root", default="outputs/crops")
    parser.add_argument("--output_dir", default="outputs/embeddings/uni2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--resize_to", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--encoder_name", default="uni2-h")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = UNICropDataset(args.crop_root, args.split, args.patch_size, args.mode, args.resize_to)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_with_metadata)
    encoder = UNI2FeatureExtractor(args.encoder_name, device=device)
    save_uni_features(encoder, dataloader, device, args.output_dir, args.split, args.mode)


if __name__ == "__main__":
    main()
