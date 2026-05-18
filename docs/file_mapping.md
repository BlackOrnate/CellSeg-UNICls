# Refactoring map

This package reorganizes the original flat scripts into the README-style project layout.

| Original location | New location | Purpose |
|---|---|---|
| `dataset.py` | `src/data/dataset.py` | Dataset classes for instance crops and saved UNI2 embeddings. |
| `models/classifier.py` | `src/classification/mlp_classifier.py` | Lightweight MLP classifier: `1536 -> 512 -> 128 -> 32 -> C`. |
| `train_classifier.py` | `scripts/train_classifier.py` | Train the classifier from saved UNI2 embeddings. |
| `test_classifier.py` | `scripts/infer.py` | Run classifier inference and save per-instance JSON predictions. |
| `tools_classifier.py` | `src/classification/uni2_encoder.py`, `src/evaluation/metrics.py`, `src/evaluation/visualize.py`, `src/utils/io.py` | Feature extraction, metric computation, visualization, and result saving utilities. |
| `train_cellpose_sam.py` | `scripts/train_cellpose_sam.py`, `src/segmentation/cellpose_sam_train.py` | Cellpose-SAM training wrapper without hard-coded local paths. |
| `test_cellpose_sam.py` | `scripts/infer_segmentation.py`, `scripts/evaluate.py` | Segmentation inference and evaluation wrappers. |
| `tools_cellpose_sam.py` | `src/segmentation/postprocess.py`, `src/evaluation/matching.py`, `src/evaluation/metrics.py`, `src/evaluation/visualize.py` | Instance metadata, optional morphology-aware merging, pairing, metrics, and overlays. |
| `docs/figures/` | `docs/figures/` | README figures. |

The original repository also contains vendored Cellpose/UNI code under `models/`. In this cleaned structure, Cellpose is loaded from the `cellpose` pip package, and UNI2 loading is wrapped in `src/classification/uni2_encoder.py` so you can plug in your local UNI2 loader or checkpoint.
