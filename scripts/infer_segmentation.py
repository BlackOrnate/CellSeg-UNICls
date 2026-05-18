from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse

from src.segmentation.inference import infer_cellpose_sam


def main():
    parser = argparse.ArgumentParser(description="Run Cellpose-SAM inference and save instance maps.")
    parser.add_argument("--image_dir", default="data/processed/images")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="outputs/predictions/instances")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    infer_cellpose_sam(args.image_dir, args.checkpoint, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
