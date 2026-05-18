from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Create a simple metadata.csv from raw images and annotations.")
    parser.add_argument("--image_dir", default="data/raw/images")
    parser.add_argument("--annotation_dir", default="data/raw/annotations")
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        parts = image_path.stem.split("_")
        rows.append({
            "image_id": image_path.stem,
            "image_path": str(image_path),
            "annotation_path": str(Path(args.annotation_dir) / f"{image_path.stem}.geojson"),
            "patient_id": parts[0] if parts else "unknown",
            "region": next((p for p in parts if p.upper() in {"CA1", "CA2", "CA3", "FIMBRIA", "DG"}), "unknown"),
        })
    pd.DataFrame(rows).to_csv(out_dir / "metadata.csv", index=False)
    print(f"Saved metadata: {out_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
