from __future__ import annotations

from pathlib import Path

import pandas as pd


def create_leave_one_patient_out_splits(
    metadata_csv: str | Path,
    output_dir: str | Path,
    patient_col: str = "patient_id",
    sample_col: str = "image_id",
) -> None:
    df = pd.read_csv(metadata_csv)
    if patient_col not in df.columns:
        raise ValueError(f"metadata file must contain `{patient_col}`")
    if sample_col not in df.columns:
        raise ValueError(f"metadata file must contain `{sample_col}`")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold_idx, patient_id in enumerate(sorted(df[patient_col].dropna().unique()), start=1):
        test_df = df[df[patient_col] == patient_id]
        train_df = df[df[patient_col] != patient_id]
        (output_dir / f"fold_{fold_idx}_train.txt").write_text(
            "\n".join(map(str, train_df[sample_col].tolist())) + "\n", encoding="utf-8"
        )
        (output_dir / f"fold_{fold_idx}_test.txt").write_text(
            "\n".join(map(str, test_df[sample_col].tolist())) + "\n", encoding="utf-8"
        )
