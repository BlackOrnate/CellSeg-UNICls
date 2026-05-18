# CellSeg-UNICls

CellSeg-UNICls is a decoupled framework for cell segmentation and classification in brain histopathology images under limited annotation conditions.

The framework combines Cellpose-SAM-based instance segmentation with UNI2 embedding-based cell classification. It is designed for brain histopathology images where dense cell-level annotations are limited and where domain shift from commonly used cancer histology datasets can reduce model performance.

## Overview

Cell-level analysis in brain histopathology is challenging because of limited annotations, complex tissue structures, and large morphological differences across brain regions. CellSeg-UNICls addresses this problem by separating the task into two stages:

1. **Instance Segmentation**  
   Cellpose-SAM is used to generate cell instance masks.

2. **Cell Type Classification**  
   For each detected instance, an instance-centered image crop is extracted and encoded using a frozen UNI2 encoder. A lightweight MLP classifier is then trained to predict the cell type.

This decoupled design allows the segmentation and classification components to be optimized independently, improving flexibility and robustness in low-data settings.

## Key Features

- Brain histopathology cell segmentation and classification
- Decoupled segmentation-classification pipeline
- Cellpose-SAM-based instance segmentation
- UNI2 embedding-based cell type classification
- Lightweight MLP classifier
- Support for limited annotation settings
- Evaluation with Dice, bPQ, mPQ, detection F1, weighted F1, and macro F1
- Optional morphology-aware post-processing for over-segmentation analysis

## Method

The overall pipeline consists of the following steps:

1. Input a brain histopathology image patch.
2. Use Cellpose-SAM to predict cell instance masks.
3. Extract instance-centered RGB crops from the original image.
4. Resize each crop to the UNI2 input resolution.
5. Extract image embeddings using a frozen UNI2 encoder.
6. Train an MLP classifier for cell type prediction.
7. Combine segmentation masks and predicted cell labels to produce the final cell-level output.

## Dataset

The project is developed and evaluated on a curated brain histopathology dataset from the NYBB brain slide collection. The dataset includes multiple hippocampal-related regions, including:

- Cornu Ammonis area 1 (CA1)
- Cornu Ammonis areas 2 and 3 (CA2/3)
- Hippocampal fimbria (Fimbria)
- Dentate gyrus

The annotated cell categories include:

- Neuron
- Oligodendrocyte
- Blood vessel region
- Debris
- Fimbria boundary cells
- Unidentifiable

The `unidentifiable` label is used for ambiguous or low-confidence instances during annotation, but it is excluded from classification training and classification metric computation.

> Note: The dataset is not included in this repository due to privacy, access, and data-sharing restrictions.

## Installation

```bash
conda create -n cellseg-unicls python=3.10
conda activate cellseg-unicls

pip install -r requirements.txt
```

## Usage

### 1. Train or fine-tune the segmentation model

```bash

```

### 2. Extract instance-centered crops

```bash

```

### 3. Extract UNI2 embeddings

```bash

```

### 4. Train the MLP classifier

```bash

```

### 5. Evaluate the full pipeline

```bash

```

## Evaluation Metrics

The framework is evaluated using both segmentation and classification metrics:

- Dice coefficient
- Binary Panoptic Quality
- Multi-class Panoptic Quality
- Detection F1 score
- Weighted classification F1 score
- Macro classification F1 score

Classification metrics are computed only on paired predicted and ground-truth instances. Unpaired predictions affect detection performance but are not included in classification F1 computation. Unidentifiable instances are excluded from classification training and evaluation.

## Results

CellSeg-UNICls maintains strong segmentation performance while improving class-aware panoptic quality and balanced classification performance. Compared with hybrid baselines, the framework shows stronger macro F1 performance, especially for minority and structurally challenging cell categories.

<div align="center">

| Model | Dice | bPQ | mPQ | F1(Det) | F1(Cls, W) | F1(Cls, M) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CellViT (SAM-H) | 0.8154 | 0.6529 | 0.4061 | 0.7898 | 0.9356 | 0.4766 |
| CellViT++ (SAM-H) | 0.8154 | 0.6529 | 0.2848 | 0.7898 | 0.6057 | 0.2895 |
| PointFormer | 0.8074 | 0.5939 | 0.3064 | 0.7816 | 0.8593 | 0.6766 |
| Cellpose-SAM | **0.8633** | **0.7834** | - | **0.8700** | - | - |
| Cellpose-SAM + CellViT cls | **0.8633** | **0.7834** | 0.5266 | **0.8700** | **0.9394** | 0.5740 |
| CellSeg-UNICls (ours) | **0.8633** | **0.7834** | **0.5548** | **0.8700** | 0.9326 | **0.7472** |

</div>

## Notes

This repository focuses on the implementation of the CellSeg-UNICls framework. The original histopathology images and annotations are not publicly released in this repository.

Before using this code on a new dataset, please update the configuration file and ensure that the input image, annotation, and label formats match the expected structure.

## Citation

If you use this repository or find this project helpful, please cite:

```text
Zhu, Hongrui. CellSeg-UNICls: Decoupled Segmentation and Embedding-based Cell Classification for Brain Histopathology. Master's Thesis, Stony Brook University, 2026.
```

## Author

Hongrui Zhu  
M.S. in Biomedical Informatics  
Stony Brook University

## License

This project is intended for academic and research use. Please check the license file for details.
