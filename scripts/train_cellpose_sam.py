from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse

from src.segmentation.cellpose_sam_train import train_cellpose_sam


def main():
    parser = argparse.ArgumentParser(description="Train Cellpose-SAM segmentation backbone.")
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dir", default=None)
    parser.add_argument("--model_name", default="cellseg_unicls_cellpose_sam")
    parser.add_argument("--checkpoint_dir", default="outputs/checkpoints/segmentation")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    args = parser.parse_args()
    train_cellpose_sam(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    main()
