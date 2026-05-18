from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

import numpy as np

from src.evaluation.metrics import evaluate_segmentation_pair
from src.utils.io import load_array, save_json


def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted instance maps against ground truth maps.")
    parser.add_argument("--pred_instance_dir", default="outputs/predictions/instances")
    parser.add_argument("--gt_instance_dir", default="data/processed/instance_maps")
    parser.add_argument("--output", default="outputs/metrics.json")
    parser.add_argument("--pairing_radius", type=float, default=12.0)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    args = parser.parse_args()

    pred_dir = Path(args.pred_instance_dir)
    gt_dir = Path(args.gt_instance_dir)
    metrics = {}
    for pred_path in sorted(pred_dir.glob("*.npy")):
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            continue
        pred = load_array(pred_path).astype(np.int32)
        gt = load_array(gt_path).astype(np.int32)
        metrics[pred_path.stem] = evaluate_segmentation_pair(pred, gt, args.pairing_radius, args.iou_threshold)

    summary = {}
    if metrics:
        keys = list(next(iter(metrics.values())).keys())
        summary = {k: float(np.mean([m[k] for m in metrics.values()])) for k in keys}
    save_json({"per_image": metrics, "summary": summary}, args.output)
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
