from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse

from src.data.splits import create_leave_one_patient_out_splits


def main():
    parser = argparse.ArgumentParser(description="Create leave-one-patient-out split files.")
    parser.add_argument("--metadata", default="data/processed/metadata.csv")
    parser.add_argument("--output_dir", default="data/splits")
    parser.add_argument("--patient_col", default="patient_id")
    parser.add_argument("--sample_col", default="image_id")
    args = parser.parse_args()
    create_leave_one_patient_out_splits(args.metadata, args.output_dir, args.patient_col, args.sample_col)
    print(f"Saved splits to {args.output_dir}")


if __name__ == "__main__":
    main()
