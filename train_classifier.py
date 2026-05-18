import torch
import os
from tools_classifier import create_uni_dataloader, create_classifier_dataloader, save_uni_features
from models.classifier import Classifier
from models.uni import get_encoder
from tqdm import tqdm
from loss import loss_fn


def check_uni_embedding_results(save_path, group, device, patch_size, resize):
    if not os.path.exists(os.path.join(save_path, "uni_embedding_results_train", group)):
        UNI_model, transform = get_encoder(enc_name='uni2-h', device=device)
        UNI_model.eval()
        for p in UNI_model.parameters():
            p.requires_grad = False
        uni_dataset_path = "/data/hongrui/Cellpose-SAM/pred_cell_crops"
        uni_dataloader = create_uni_dataloader(uni_dataset_path, 1000, "train", "train", patch_size, resize)
        save_uni_features(UNI_model, uni_dataloader, device, save_path, "train", "train")
        uni_dataloader = create_uni_dataloader(uni_dataset_path, 1000, "test", "train", patch_size, resize)
        save_uni_features(UNI_model, uni_dataloader, device, save_path, "test", "train")


def train_classifier(save_path, epoch_size, device):
    classifier_model = Classifier(num_classes=7)
    optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",  # 如果监控 macro F1
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    classifier_dataloader_train = create_classifier_dataloader(save_path, 1000, "train", "train")
    classifier_dataloader_test = create_classifier_dataloader(save_path, 1000, "test", "train")
    min_val_loss = 99999

    for epoch_num in range(epoch_size):
        classifier_model.train()
        classifier_model.to(device)
        total_loss = 0.0
        for feats, types, path_names, cell_ids in tqdm(classifier_dataloader_train,
                                                       total=len(classifier_dataloader_train),
                                                       desc=f"Training Epoch {epoch_num + 1}"):
            type_pred = classifier_model(feats.to(device))
            loss = loss_fn(type_pred, types.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # save_feat(feats, path_names, cell_ids, save_path)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch_num + 1}, train_loss={total_loss / len(classifier_dataloader_train):.4f}, lr={current_lr:.4f}")

        classifier_model.eval()
        classifier_model.to(device)
        total_loss = 0.0
        for feats, types, path_names, cell_ids in tqdm(classifier_dataloader_test,
                                                       total=len(classifier_dataloader_test),
                                                       desc=f"Evaluating Epoch {epoch_num + 1}"):
            type_pred = classifier_model(feats.to(device))
            loss = loss_fn(type_pred, types.to(device))
            total_loss += loss.item()

            # save_feat(feats, path_names, cell_ids, save_path)
        print(f"Epoch {epoch_num + 1}, test_loss={total_loss / len(classifier_dataloader_test):.4f}")
        scheduler.step(total_loss)

        if min_val_loss > total_loss:
            min_val_loss = total_loss
            torch.save(classifier_model.state_dict(), os.path.join(save_path, "classifier_model.pth"))


if __name__ == '__main__':
    gpu_id = 2

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    group = "train"
    patch_size = 64
    resize = True
    save_path = "/data/hongrui/UNI/results"
    epoch_size = 100

    check_uni_embedding_results(save_path, group, device, patch_size, resize)

    train_classifier(save_path, epoch_size, device)
