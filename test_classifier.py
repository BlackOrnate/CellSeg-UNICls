import torch
import os
from tools_classifier import create_uni_dataloader, create_classifier_dataloader, save_uni_features, result_to_dict, \
    save_uni_results
from models.classifier import Classifier
from models.uni import get_encoder
from tqdm import tqdm
from loss import loss_fn
import torch.nn.functional as F


def check_uni_embedding_results(save_path, group, device, patch_size, resize):
    if not os.path.exists(os.path.join(save_path, "uni_embedding_results_test", group)):
        UNI_model, transform = get_encoder(enc_name='uni2-h', device=device)
        UNI_model.eval()
        for p in UNI_model.parameters():
            p.requires_grad = False
        uni_dataset_path = "/data/hongrui/Cellpose-SAM/pred_cell_crops"
        uni_dataloader = create_uni_dataloader(uni_dataset_path, 1000, "test", "test", patch_size, resize)
        save_uni_features(UNI_model, uni_dataloader, device, save_path, "test", "test")


def test_classifier(save_path, device):
    classifier_model = Classifier(num_classes=7)
    classifier_dataloader_test = create_classifier_dataloader(save_path, 1000, "test", "test")
    classifier_model.eval()
    classifier_model.to(device)
    classifier_model.load_state_dict(torch.load(os.path.join(save_path, "classifier_model(new)(dropout).pth")))

    gt_types = []
    pred_types = []
    uni_result_dict = {}
    for feats, types, path_names, cell_ids in tqdm(classifier_dataloader_test, total=len(classifier_dataloader_test),
                                                   desc=f"Evaluating"):
        type_gt_values = types.cpu().numpy()
        type_pred = classifier_model(feats.to(device))

        type_pred_prob = F.softmax(type_pred, dim=1).detach().cpu().numpy()
        type_pred_values = type_pred.argmax(dim=1).cpu().numpy() + 1  # [B]

        gt_types.extend(type_gt_values)
        pred_types.extend(type_pred_values)

        uni_result_dict = result_to_dict(uni_result_dict, path_names, cell_ids, type_pred_values, type_pred_prob)

    # calculate_original_matrics(patch_size, group, save_path)
    # calculate_new_matrices(gt_types, pred_types, save_path)
    save_uni_results(save_path, uni_result_dict)


if __name__ == '__main__':
    gpu_id = 2

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    group = "test"
    patch_size = 64
    resize = True
    save_path = "/data/hongrui/UNI/results"

    check_uni_embedding_results(save_path, group, device, patch_size, resize)

    test_classifier(save_path, device)
