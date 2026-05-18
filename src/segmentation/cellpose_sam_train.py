from __future__ import annotations

from pathlib import Path


def train_cellpose_sam(
    train_dir: str | Path,
    test_dir: str | Path | None,
    model_name: str,
    checkpoint_dir: str | Path,
    epochs: int = 100,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.1,
):
    """Train Cellpose-SAM using the Cellpose training API.

    This removes the hard-coded local paths from the original root-level script.
    The exact Cellpose API can differ by version, so this wrapper may need small edits if your
    local Cellpose-SAM package exposes `train_seg` under a different module.
    """
    try:
        from cellpose import core, io, models, train
    except Exception as exc:
        raise RuntimeError("Please install Cellpose/Cellpose-SAM before training segmentation.") from exc

    if not core.use_gpu():
        raise RuntimeError("No GPU detected by Cellpose. Please check CUDA/Cellpose installation.")

    io.logger_setup()
    output = io.load_train_test_data(str(train_dir), str(test_dir) if test_dir else None)
    train_data, train_labels, train_type_maps, train_img_names, test_data, test_labels, test_type_maps, test_img_names = output

    model = models.CellposeModel(gpu=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    new_model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_data,
        train_labels=train_labels,
        train_type_maps=train_type_maps,
        test_data=test_data,
        test_labels=test_labels,
        batch_size=batch_size,
        n_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        nimg_per_epoch=max(2, len(train_data)),
        model_name=model_name,
    )
    return new_model_path, train_losses, test_losses
