import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torch
from torchvision import transforms


class UNIDataset(Dataset):
    def __init__(self, dataset_path, group, patch_size, used, resize=True):
        super().__init__()
        self.dataset_path = dataset_path
        self.group = group
        self.image_paths = []
        self.cell_ids = []
        self.types = []
        self.path_names = []

        for path_name in os.listdir(f"{self.dataset_path}/{patch_size}/{self.group}"):
            df = pd.read_csv(f"{self.dataset_path}/{patch_size}/{self.group}/{path_name}/pred_cells_gt_type.csv")
            for file_name in os.listdir(f"{self.dataset_path}/{patch_size}/{self.group}/{path_name}"):
                if file_name.endswith(".png"):
                    cell_id = int(file_name.split(".")[0])
                    matched_gt_type = int(df[df["pred_cell_id"] == cell_id]["matched_gt_type"].values[0])
                    if used == "train":
                        if matched_gt_type != -1:
                            self.cell_ids.append(cell_id)
                            self.image_paths.append(
                                f"{self.dataset_path}/{patch_size}/{self.group}/{path_name}/{file_name}")
                            self.types.append(matched_gt_type)
                            self.path_names.append(path_name)
                    elif used == "test":
                        self.cell_ids.append(cell_id)
                        self.image_paths.append(
                            f"{self.dataset_path}/{patch_size}/{self.group}/{path_name}/{file_name}")
                        self.types.append(matched_gt_type)
                        self.path_names.append(path_name)

        transform_list = []

        if resize:
            transform_list.append(transforms.Resize((224, 224)))

        transform_list.append(transforms.ToTensor())

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        type = self.types[idx]
        path_name = self.path_names[idx]
        cell_id = self.cell_ids[idx]

        return image, type, path_name, cell_id


class ClassifierDataset(Dataset):
    def __init__(self, dataset_path, group, used):
        super().__init__()
        self.feature_paths = []

        split_root = os.path.join(dataset_path,
                                  "uni_embedding_results_train" if used == "train" else "uni_embedding_results_test",
                                  group)
        for path_name in os.listdir(split_root):
            path_folder = os.path.join(split_root, path_name)
            if not os.path.isdir(path_folder):
                continue

            for file_name in os.listdir(path_folder):
                if file_name.endswith(".pt"):
                    self.feature_paths.append(os.path.join(path_folder, file_name))

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        data = torch.load(self.feature_paths[idx])

        feat = data["feat"].float()
        label = int(data["label"])
        path_name = data["path_name"]
        cell_id = int(data["cell_id"])

        return feat, label, path_name, cell_id
