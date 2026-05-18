import numpy as np
from models.cellpose import models, core, io
from pathlib import Path
from tqdm import trange
from models.cellpose import metrics

from tools_cellpose_sam import unpack_predictions, unpack_ground_truth, calculate_step_metric, plot_results, show_metrics, \
    calculate_step_metric_with_class, show_metrics_with_class

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

io.logger_setup()  # run this to get printing of progress

# Check if colab notebook instance has GPU access
if core.use_gpu() == False:
    raise ImportError("No GPU access, change your runtime")

model_name = "230_5"

type_map_path = "/data/hongrui/NYBB/CLI/logs/2026-02-02T205859_Train/full_inference_predictions/type_map"
uni_result_path = "/data/hongrui/UNI/results/uni_results"

# default training params
n_epochs = 100
learning_rate = 1e-5
weight_decay = 0.1
batch_size = 1

run_dir = "/data/hongrui/Cellpose-SAM/result"
plt_output_path = "/data/hongrui/Cellpose-SAM/result"
train_dir = f"/data/hongrui/NYBB/dataset/patients/{model_name}/fold0"
test_dir = f"/data/hongrui/NYBB/dataset/patients/{model_name}/fold1"

# get files
output = io.load_train_test_data(train_dir, test_dir)
train_data, train_labels, train_type_maps, train_img_names, test_data, test_labels, test_type_maps, test_img_names = output

new_model_path = Path(f"/data/hongrui/Cellpose-SAM/models/{model_name}")

model = models.CellposeModel(gpu=True,
                             pretrained_model=new_model_path)

masks, flows, styles = model.eval(test_data, batch_size=32)
pred = unpack_predictions(masks, test_img_names, uni_result_path, type_map_path, styles)
gt = unpack_ground_truth(test_labels, test_img_names, type_map_path, test_type_maps)

if type_map_path:
    batch_metrics = calculate_step_metric_with_class(gt, pred, test_img_names, input_imgs=test_data, crop_size=56)
    show_metrics_with_class(batch_metrics, run_dir, model_name)
else:
    batch_metrics = calculate_step_metric(gt, pred, test_img_names)
    show_metrics(batch_metrics, run_dir, model_name)

plot_results(test_data, pred, gt, test_img_names, plt_output_path)

# check performance using ground truth labels
ap = metrics.average_precision(test_labels, masks)[0]
print('')
print(f'>>> average precision at iou threshold 0.5 = {ap[:, 0].mean():.3f}')
