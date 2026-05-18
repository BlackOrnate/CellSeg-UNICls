from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classification.losses import classification_loss
from src.classification.mlp_classifier import MLPClassifier
from src.data.dataset import UNIEmbeddingDataset, collate_with_metadata
from src.utils.io import ensure_dir
from src.utils.seed import seed_everything


def run_epoch(model, dataloader, optimizer, device, label_offset, ignore_label, train=True):
    model.train(train)
    total = 0.0
    for feats, labels, _, _ in tqdm(dataloader, total=len(dataloader), desc="train" if train else "val"):
        feats = feats.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(train):
            logits = model(feats)
            loss = classification_loss(logits, labels, label_offset=label_offset, ignore_label=ignore_label)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += float(loss.item())
    return total / max(len(dataloader), 1)


def main():
    parser = argparse.ArgumentParser(description="Train UNI2-embedding MLP classifier.")
    parser.add_argument("--embedding_root", default="outputs/embeddings/uni2")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="test")
    parser.add_argument("--checkpoint_dir", default="outputs/checkpoints/classification")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--label_offset", type=int, default=1)
    parser.add_argument("--ignore_label", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = UNIEmbeddingDataset(args.embedding_root, args.train_split, mode="train")
    val_ds = UNIEmbeddingDataset(args.embedding_root, args.val_split, mode="train")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_with_metadata)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_with_metadata)

    model = MLPClassifier(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    ckpt_dir = ensure_dir(args.checkpoint_dir)
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, args.label_offset, args.ignore_label, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, args.label_offset, args.ignore_label, train=False)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_dir / "classifier_model.pth")
            print(f"Saved best checkpoint to {ckpt_dir / 'classifier_model.pth'}")


if __name__ == "__main__":
    main()
