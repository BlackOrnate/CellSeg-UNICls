from models.cellpose import train
import numpy as np
from models.cellpose import models, core, io
from tqdm import trange

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

io.logger_setup()  # run this to get printing of progress

# a = np.load("/data/hongrui/Cellpose-SAM/dataset/human_in_the_loop/test/breast_vectra_0_seg.npy", allow_pickle=True)

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
# (not passing test data into function to speed up training)

model = models.CellposeModel(gpu=True)
new_model_path, train_losses, test_losses = train.train_seg(model.net,
                                                            train_data=train_data,
                                                            train_labels=train_labels,
                                                            train_type_maps=train_type_maps,
                                                            batch_size=batch_size,
                                                            n_epochs=n_epochs,
                                                            learning_rate=learning_rate,
                                                            weight_decay=weight_decay,
                                                            nimg_per_epoch=max(2, len(train_data)),  # can change this
                                                            model_name=model_name)
