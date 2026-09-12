
# Breast Cancer Histopathological Image Classification

A deep learning project for **binary classification of breast histopathological image patches** using **Transfer Learning with ResNet50V2**.

The project explores a semi-supervised/weakly supervised-style classification setup on histopathological images and aims to classify image samples into two categories:

* **No** — No breast cancer
* **Yes** — Breast cancer

> **Author:** Mir Siam
> **Student ID:** 213-15-4438
> **Program:** CSE

---

## 📌 Project Overview

Histopathological image analysis is an important application of computer vision in medical image analysis. This project applies a convolutional neural network based approach to classify breast histopathological image patches into two classes.

Instead of training a deep CNN entirely from scratch, the project uses **ResNet50V2 pretrained on ImageNet** as a feature extractor. The pretrained backbone is initially frozen, and a lightweight classification head is added on top.

The implemented architecture consists of:

```text
Input Image
    │
    ▼
224 × 224 × 3
    │
    ▼
ResNet50V2
(ImageNet Pretrained)
    │
    ▼
Global Average Pooling
    │
    ▼
Dropout (0.2)
    │
    ▼
Dense Layer
1 Neuron + Sigmoid
    │
    ▼
Binary Prediction
No / Yes
```

The ResNet50V2 backbone produces a `7 × 7 × 2048` feature representation, followed by Global Average Pooling and a single sigmoid output neuron.

---

## 🎯 Objectives

The main objectives of this project are:

* Perform binary classification of breast histopathological images.
* Explore deep learning for medical image classification.
* Utilize **transfer learning** to leverage pretrained visual features.
* Build a lightweight classification head on top of ResNet50V2.
* Analyze the dataset and its class distribution.
* Train and validate a binary image classification model.
* Save the best-performing model during training.

---

## 🧬 Dataset

The dataset consists of histopathological image patches organized into two directories:

```text
10264/
├── 0/
│   └── No breast cancer images
│
└── 1/
    └── Breast cancer images
```

The original notebook loads the dataset from Google Drive and maps the two folders to the labels `No` and `Yes`.

### Dataset Statistics

| Class            | Samples |
| ---------------- | ------: |
| No Breast Cancer |     617 |
| Breast Cancer    |     103 |
| **Total**        | **720** |

The dataset is therefore **imbalanced**, with substantially more samples in the `No` class than the `Yes` class.

### Data Split

The notebook uses:

| Dataset    |  Images |
| ---------- | ------: |
| Training   |     615 |
| Validation |      69 |
| Testing    |      36 |
| **Total**  | **720** |

The generators report two classes for each split.

The class mapping used by the generator is:

```python
{
    'No': 0,
    'Yes': 1
}
```

---

## 🧠 Model Architecture

### ResNet50V2

The project uses:

```python
keras.applications.ResNet50V2(
    weights="imagenet",
    input_shape=(224, 224, 3),
    include_top=False
)
```

The pretrained ResNet50V2 classification head is removed, allowing the network to act as a feature extractor.

The base model is initially frozen:

```python
base_model.trainable = False
```

This means the pretrained ResNet50V2 weights are not updated during the initial training stage.

### Classification Head

A custom classification head is placed on top of ResNet50V2:

```python
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)
```

The final sigmoid neuron produces a probability for binary classification.

### Model Parameters

The resulting model contains:

| Parameter Type           |      Count |
| ------------------------ | ---------: |
| Total Parameters         | 23,566,849 |
| Trainable Parameters     |      2,049 |
| Non-trainable Parameters | 23,564,800 |

The vast majority of parameters belong to the frozen ResNet50V2 backbone.

---

## ⚙️ Training Configuration

The model is compiled using:

```python
model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=0.0001),
    metrics=["accur]()
```
