import os
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from dataset import UNIDataset, ClassifierDataset
import matplotlib.pyplot as plt
import torch
import json


def create_uni_dataloader(dataset_path, batch_size, group, used, patch_size, resize):
    dataset = UNIDataset(dataset_path, group, patch_size, used, resize)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2,
                      collate_fn=collate_fn)


def create_classifier_dataloader(dataset_path, batch_size, group, used):
    dataset = ClassifierDataset(dataset_path, group, used)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2,
                      collate_fn=collate_fn)


def collate_fn(batch):
    images, types, path_names, cell_ids = zip(*batch)

    images = torch.stack(images, dim=0)
    types = torch.tensor(types, dtype=torch.long)
    path_names = list(path_names)
    cell_ids = list(cell_ids)

    return images, types, path_names, cell_ids


def save_uni_features(uni_model, dataloader, device, save_root, split, used):
    uni_model.eval()
    os.makedirs(os.path.join(save_root, split), exist_ok=True)

    with torch.no_grad():
        for images, types, path_names, cell_ids in tqdm(
                dataloader, total=len(dataloader), desc=f"Extracting UNI features ({split})"
        ):
            images = images.to(device)
            feats = uni_model(images)  # [B, 1536]

            feats = feats.cpu()

            for feat, label, path_name, cell_id in zip(feats, types, path_names, cell_ids):
                folder = os.path.join(save_root,
                                      "uni_embedding_results_train" if used == "train" else "uni_embedding_results_test",
                                      split, str(path_name))
                os.makedirs(folder, exist_ok=True)

                save_path = os.path.join(folder, f"{cell_id}.pt")
                torch.save(
                    {
                        "feat": feat,  # [1536]
                        "label": int(label),
                        "path_name": path_name,
                        "cell_id": int(cell_id),
                    },
                    save_path
                )


def evaluate_model(gt_types, pred_types, model_name, save_dir):
    gt_types = np.asarray(gt_types)
    pred_types = np.asarray(pred_types)

    # =========================
    # A. 主任务：真实细胞分类
    # 忽略 gt==0(non-cell) 和 gt==1(ignore)
    # =========================
    mask_main = (gt_types != -1) & (gt_types != 1)
    y_true_main = gt_types[mask_main]
    y_pred_main = pred_types[mask_main]

    calculate_metrics(
        gt_types=y_true_main,
        pred_types=y_pred_main,
        labels=np.arange(2, 7),
        class_names=["Neuron", "Oligo", "?? (debris)", "Blood Vessel", "Fimbria"],
        flag="Main_Cell_Classification",
        model_name=model_name,
        save_dir=save_dir,
        plot_bar=True
    )

    # =========================
    # B. 包含 non-cell 的多分类评估
    # 忽略 gt==1
    # =========================
    mask_with_noncell = gt_types != 1
    y_true_with_noncell = gt_types[mask_with_noncell]
    y_pred_with_noncell = pred_types[mask_with_noncell]

    calculate_metrics(
        gt_types=y_true_with_noncell,
        pred_types=y_pred_with_noncell,
        labels=np.arange(0, 7),
        class_names=["Background", "Unidentifiable", "Neuron", "Oligo", "?? (debris)", "Blood Vessel", "Fimbria"],
        flag="With_NonCell",
        model_name=model_name,
        save_dir=save_dir,
        plot_bar=False
    )

    # =========================
    # C. cell vs non-cell 二分类评估
    # 忽略 gt==1
    # 0 -> non-cell
    # 2~7 -> cell
    # =========================
    mask_binary = gt_types != 1
    y_true_binary = (gt_types[mask_binary] != 0).astype(int)  # 1=cell, 0=non-cell
    y_pred_binary = (pred_types[mask_binary] != 0).astype(int)

    calculate_metrics(
        gt_types=y_true_binary,
        pred_types=y_pred_binary,
        labels=np.arange(0, 2),
        class_names=["Non-cell", "Cell"],
        flag="Cell_vs_NonCell",
        model_name=model_name,
        save_dir=save_dir,
        plot_bar=False
    )


def calculate_original_matrics(patch_size, group, save_dir):
    dataset_path = "/data/hongrui/Cellpose-SAM/pred_cell_crops"
    all_y_true = []
    all_y_pred = []
    for path_name in os.listdir(f"{dataset_path}/{patch_size}/{group}"):
        csv_path = f"{dataset_path}/{patch_size}/{group}/{path_name}/pred_cells_gt_type.csv"
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        df = df[df["matched_gt_type"] != -1].copy()

        if len(df) == 0:
            continue

        all_y_true.extend(df["matched_gt_type"].astype(int).tolist())
        all_y_pred.extend(df["pred_type"].astype(int).tolist())

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    evaluate_model(
        gt_types=all_y_true,
        pred_types=all_y_pred,
        model_name="OldModel",
        save_dir=save_dir
    )


def calculate_new_matrices(gt_types, pred_types, save_dir):
    evaluate_model(
        gt_types=gt_types,
        pred_types=pred_types,
        model_name="UNI2_Classifier",
        save_dir=save_dir
    )


def calculate_metrics(gt_types, pred_types, labels, class_names, flag, model_name, save_dir, plot_bar):
    acc = accuracy_score(gt_types, pred_types)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        gt_types, pred_types, labels=labels, average="macro", zero_division=0
    )

    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        gt_types, pred_types, labels=labels, average="weighted", zero_division=0
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        gt_types, pred_types, labels=labels, average=None, zero_division=0
    )

    print("~" * 80)
    print(f"[{model_name} | {flag}] Acc : {acc:.4f}")
    print(
        f"[{model_name} | {flag}] Macro    - Precision: {prec_macro:.4f}, Recall: {rec_macro:.4f}, F1: {f1_macro:.4f}")
    print(f"[{model_name} | {flag}] Weighted - Precision: {prec_w:.4f}, Recall: {rec_w:.4f}, F1: {f1_w:.4f}")

    print("\nPer-class metrics:")
    print("-" * 80)
    for i, cls in enumerate(labels):
        print(
            f"[{model_name} | {flag}] Class {cls} ({class_names[i]}): "
            f"Precision: {precision[i]:.4f}, "
            f"Recall: {recall[i]:.4f}, "
            f"F1: {f1[i]:.4f}, "
            f"N: {support[i]}"
        )

    # plot_raw_confusion_matrix(
    #     gt_types=gt_types,
    #     pred_types=pred_types,
    #     labels=labels,
    #     class_names=class_names,
    #     flag=flag,
    #     model_name=model_name,
    #     save_dir=save_dir
    # )

    plot_normalized_confusion_matrix(
        gt_types=gt_types,
        pred_types=pred_types,
        labels=labels,
        class_names=class_names,
        flag=flag,
        model_name=model_name,
        save_dir=save_dir
    )

    if plot_bar:
        plot_per_class_bar(
            precision=precision,
            recall=recall,
            f1=f1,
            class_names=class_names,
            flag=flag,
            model_name=model_name,
            save_dir=save_dir
        )


def plot_raw_confusion_matrix(gt_types, pred_types, labels, class_names, flag, model_name, save_dir=None):
    cm = confusion_matrix(gt_types, pred_types, labels=labels, normalize=None)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)

    ax.set_title(f"{model_name} - {flag} Confusion Matrix (Raw Counts)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{model_name}_{flag}_confusion_matrix_raw.png"), dpi=300)

    plt.show()
    plt.close()


def plot_normalized_confusion_matrix(gt_types, pred_types, labels, class_names, flag, model_name, save_dir=None):
    cm = confusion_matrix(gt_types, pred_types, labels=labels, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=True)

    ax.set_title(f"{model_name} - {flag} Normalized Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{model_name}_{flag}_normalized_confusion_matrix.png"), dpi=300)

    plt.show()
    plt.close()


def plot_per_class_bar(precision, recall, f1, class_names, flag, model_name, save_dir=None):
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1-score")

    ax.set_xlabel("Type")
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name} - {flag} Per-class Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_ylim(0, 1.05)
    ax.legend()

    # 在柱子上显示数值（可选）
    for i, v in enumerate(precision):
        ax.text(i - width, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(recall):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(f1):
        ax.text(i + width, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{model_name}_{flag}_per_class_metrics_bar.png"), dpi=300)

    plt.show()
    plt.close()


def result_to_dict(uni_result_dict, path_names, cell_ids, type_pred_values, type_pred_probs):
    for i in range(len(type_pred_values)):
        path_name = path_names[i]
        cell_id = cell_ids[i]
        type_pred_prob = type_pred_probs[i]
        type_pred_value = type_pred_values[i]

        if path_name not in uni_result_dict.keys():
            uni_result_dict[path_name] = {}

        uni_result_dict[path_name][cell_id] = {
            "type": int(type_pred_value),
            "type_prob": float(type_pred_prob[type_pred_value - 1]),
        }

    return uni_result_dict


def save_uni_results(base_save_path, uni_result_dict):
    for patch_name in uni_result_dict.keys():
        save_path = f"{base_save_path}/uni_results/{patch_name}.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w", encoding='utf-8') as f:
            json.dump(uni_result_dict[patch_name], f, indent=4, ensure_ascii=False)
            f.close()
