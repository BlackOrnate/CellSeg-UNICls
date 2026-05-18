from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse

from src.classification.crop_extractor import extract_instance_crops


def main():
    parser = argparse.ArgumentParser(description="Extract instance-centered RGB crops from instance maps.")
    parser.add_argument("--image_dir", default="data/processed/images")
    parser.add_argument("--instance_dir", default="outputs/predictions/instances")
    parser.add_argument("--output_dir", default="outputs/crops")
    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument("--split", default="test")
    parser.add_argument("--labels_csv", default=None)
    args = parser.parse_args()
    extract_instance_crops(args.image_dir, args.instance_dir, args.output_dir, args.crop_size, args.split, args.labels_csv)


if __name__ == "__main__":
    main()
