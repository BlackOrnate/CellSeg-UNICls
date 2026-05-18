from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classification.mlp_classifier import MLPClassifier
from src.data.dataset import UNIEmbeddingDataset, collate_with_metadata
from src.utils.io import save_json


def main():
    parser = argparse.ArgumentParser(description="Run classifier inference and save per-patch JSON results.")
    parser.add_argument("--embedding_root", default="outputs/embeddings/uni2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/classification/classifier_model.pth")
    parser.add_argument("--output_dir", default="outputs/predictions/classification/uni_results")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--class_id_offset", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = UNIEmbeddingDataset(args.embedding_root, args.split, mode="test")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_with_metadata)
    model = MLPClassifier(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    results = defaultdict(dict)
    with torch.no_grad():
        for feats, _, patch_names, cell_ids in tqdm(loader, total=len(loader), desc="Classifier inference"):
            probs = F.softmax(model(feats.to(device)), dim=1).cpu()
            pred_idx = probs.argmax(dim=1)
            pred_type = pred_idx + args.class_id_offset
            for patch_name, cell_id, class_idx, cell_type, prob_vec in zip(patch_names, cell_ids, pred_idx, pred_type, probs):
                results[str(patch_name)][str(int(cell_id))] = {
                    "type": int(cell_type),
                    "type_prob": float(prob_vec[int(class_idx)]),
                }

    for patch_name, patch_result in results.items():
        save_json(patch_result, f"{args.output_dir}/{patch_name}.json")
    print(f"Saved classification JSON files to {args.output_dir}")


if __name__ == "__main__":
    main()
