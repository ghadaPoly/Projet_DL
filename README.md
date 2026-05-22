# Smart Waste Classification

Automated waste image classification using Transfer Learning with MobileNetV2.
6-class classifier trained on a real-world dataset, improved through iterative experimentation and offline data augmentation.

**Final model — Test Accuracy: 89.6% | Macro F1: 0.893 | Trash Recall: 88.0%**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Model Architecture](#3-model-architecture)
4. [Project Structure](#4-project-structure)
5. [Training and Experiments](#5-training-and-experiments)
6. [Results](#6-results)
7. [Application](#7-application)
8. [Technical Notes](#8-technical-notes)
9. [How to Run](#9-how-to-run)

---

## 1. Project Overview

This project builds an automatic waste classifier capable of identifying a waste item from a photo and assigning it to one of six recycling categories. The goal is to address one of the core inefficiencies in waste management: manual sorting.

**Why CNN and not MLP or RNN?**

A Multilayer Perceptron (MLP) treats each pixel as an independent value and has no notion of spatial structure. On a 224x224 image, this means 150,528 independent inputs with no understanding that adjacent pixels form edges, textures, or shapes. CNNs use convolutional filters that slide across the image and detect local patterns — edges, textures, contours — and build hierarchical representations. This is precisely what is needed to distinguish transparent glass from shiny metal or rough cardboard.

RNNs (LSTM, GRU) are designed for sequential data (text, audio, time series) and are not applicable to static image classification.

**Model choice: MobileNetV2 via Transfer Learning**

This project uses MobileNetV2, a Convolutional Neural Network pre-trained on ImageNet (1.2 million images, 1,000 classes). Instead of training from scratch on 2,527 images, Transfer Learning reuses the visual knowledge already encoded in MobileNetV2 and fine-tunes only the classification head for the 6 waste categories. This approach achieved 84.5% accuracy at baseline in approximately 6 minutes of training.

**Classes:**
`cardboard` | `glass` | `metal` | `paper` | `plastic` | `trash`

---

## 2. Dataset

**Source:** Garbage Classification Dataset — Gary Thung & Mindy Yang (GitHub)

### Initial class distribution

| Class     | Images (initial) | Images (after augmentation) |
|-----------|------------------|-----------------------------|
| cardboard | 403              | 403                         |
| glass     | 501              | 501                         |
| metal     | 410              | 410                         |
| paper     | 594              | 594                         |
| plastic   | 482              | 482                         |
| trash     | 137              | 937                         |
| **Total** | **2,527**        | **3,327**                   |

The `trash` class had only 137 images — approximately 4 times fewer than `paper`. This imbalance caused the baseline model to systematically avoid predicting `trash`, resulting in a recall of 70.4% for that class despite an acceptable global accuracy.

### Data split

| Split      | Ratio | Purpose                                         |
|------------|-------|-------------------------------------------------|
| Train      | 70%   | Model training                                  |
| Validation | 15%   | Monitor overfitting during training             |
| Test       | 15%   | Final evaluation only — never seen during training |

The test set is strictly independent. Using validation data for final evaluation would produce overly optimistic results, since the model influences validation performance indirectly through hyperparameter selection.

### Preprocessing

**Resizing to 224x224 pixels:** MobileNetV2 was designed and trained with this exact input size. All images must share identical dimensions for the network to process them.

**Normalization to [-1, 1]:** Pixel values range from 0 (black) to 255 (white). Raw values produce unstable gradients during training. MobileNetV2's `preprocess_input` applies the transformation `(pixel / 127.5) - 1`, centering values around zero. This matches the normalization used during ImageNet pre-training and is critical for correct model behavior.

**Online data augmentation (training only):** Random horizontal flip, random rotation (20%), random zoom (20%). Applied at each epoch to increase effective dataset diversity without modifying files on disk.

---

## 3. Model Architecture

### Why MobileNetV2 and not another CNN?

MobileNetV2 is built on two key innovations that make it lightweight and fast:

**Depthwise separable convolutions:** A standard convolution processes all color channels simultaneously at each position — computationally expensive. MobileNetV2 decomposes this into two steps: a depthwise convolution (one filter per channel independently) followed by a pointwise convolution (1x1) that combines channels. This reduces the number of operations by approximately 8-9x with minimal accuracy loss.

**Inverted residual blocks with skip connections:** Each block first expands the number of feature channels (x6), applies the depthwise convolution in this larger space, then compresses back to a small channel count. A skip connection adds the input directly to the output, allowing gradients to flow through without passing through every layer — preventing the gradient vanishing problem common in deep networks.

**Comparison with alternatives:**

| Model       | Parameters | Notes                                                        |
|-------------|------------|--------------------------------------------------------------|
| MobileNetV2 | 3.4M       | Chosen — lightweight, stable fine-tuning, 15 epochs in 6 min |
| VGG16       | 138M       | Too large for a 2,527-image dataset                         |
| EfficientNet | Larger    | Tested — accuracy stuck at 24% for 10 epochs (see below)    |
| MobileNetV1 | 4.2M       | No skip connections — more prone to gradient vanishing       |
| MobileNetV3 | 5.4M       | More complex operations, less stable on small datasets       |

**Why not EfficientNet?** EfficientNet was tested but produced the following results:

```
Epoch 1/10  accuracy: 0.2074  val_accuracy: 0.1802
Epoch 5/10  accuracy: 0.2134  val_accuracy: 0.2436
Epoch 10/10 accuracy: 0.2317  val_accuracy: 0.2436
```

Accuracy was stuck at approximately 24% across all 10 epochs — the model was not learning. The likely causes are (1) a preprocessing mismatch: EfficientNet requires its own `preprocess_input` function, different from MobileNetV2; using the wrong normalization sends pixels outside the range the model was trained on, preventing convergence. (2) EfficientNet is significantly larger and requires more data and a lower learning rate to converge than was available here.

### Full architecture

```
Input (224 x 224 x 3)
    |
Data Augmentation (RandomFlip, RandomRotation, RandomZoom)
    |   active during training only — Functional API ensures
    |   augmentation is disabled at inference
    |
preprocess_input (pixels normalized to [-1, 1])
    |
MobileNetV2 base (frozen at baseline / last 30 layers unfrozen at fine-tuning)
    |
GlobalAveragePooling2D
    |   reduces each feature map to a single value
    |   avoids Flatten which would create millions of parameters
    |
Dense (128, activation=ReLU)
    |
Dropout (0.3)
    |   30% of neurons randomly disabled during training
    |   prevents over-reliance on specific neurons — reduces overfitting
    |
Dense (6, activation=Softmax)
    |
Output: probability distribution over 6 classes
```

### Activation functions

| Layer           | Activation | Reason                                                         |
|-----------------|------------|----------------------------------------------------------------|
| Dense (128)     | ReLU       | Fast, avoids gradient vanishing better than sigmoid or tanh    |
| MobileNetV2 internal | ReLU6 | ReLU capped at 6 — optimized for low-precision inference    |
| Output Dense (6) | Softmax   | Converts 6 raw scores into probabilities summing to 1          |

### Training configuration

| Parameter       | Baseline        | Fine-tuning               |
|-----------------|-----------------|---------------------------|
| Optimizer       | Adam (lr=1e-3)  | Adam (lr=1e-5)            |
| Loss function   | sparse_categorical_crossentropy | sparse_categorical_crossentropy |
| Epochs          | 15              | 10                        |
| Base model      | Fully frozen    | Last 30 layers unfrozen   |

`sparse_categorical_crossentropy` is used because labels are integers (0, 1, 2, ..., 5) rather than one-hot vectors.

The fine-tuning learning rate of 1e-5 (100x smaller than the default) is essential. A standard learning rate during fine-tuning overwrites the pre-trained weights and causes catastrophic forgetting — the model loses the visual knowledge it acquired from ImageNet.

**Gradient vanishing / exploding:** MobileNetV2 is structurally designed to prevent these problems through its skip connections (gradients can bypass layers) and internal Batch Normalization (activations are kept in a stable range between each block). No divergence or stagnation was observed in the training logs.

---

## 4. Project Structure

```
smart-waste-classification/
|
|-- dataset/
|   |-- cardboard/
|   |-- glass/
|   |-- metal/
|   |-- paper/
|   |-- plastic/
|   `-- trash/               (137 original + 800 augmented images)
|
|-- model/
|   |-- waste_classifier.keras        (baseline — 84.5%)
|   |-- exp3_combined.keras           (Exp 3 V1 — before trash augmentation)
|   |-- v2_exp3_combined.keras        (Exp 3 V2 — best model — 89.6%)
|   |-- class_names.npy
|   |-- true_classes.npy
|   `-- predicted_classes.npy
|
|-- notebooks/
|   |-- data_exploration.ipynb
|   |-- training_MobileNet_CORRECTED.ipynb
|   |-- evaluation_testing.ipynb
|   |-- experiments_improvements.ipynb
|   |-- data_augmentation_trash.ipynb
|   |-- experiments_improvements_after_data_aug.ipynb
|   `-- evaluation_exp3_v2.ipynb
|
`-- app/
    `-- app.py                        (Streamlit — dual model comparison)
```

### Notebook descriptions

| Notebook | Role |
|---|---|
| `data_exploration.ipynb` | Dataset analysis, class distribution visualization, image samples |
| `training_MobileNet_CORRECTED.ipynb` | Baseline model training — Transfer Learning, 15 epochs, frozen base |
| `evaluation_testing.ipynb` | Baseline evaluation — accuracy, loss, confusion matrix, per-class F1 |
| `experiments_improvements.ipynb` | 3 improvement experiments (V1) — on the original imbalanced dataset |
| `data_augmentation_trash.ipynb` | Offline augmentation of the `trash` class (137 to 937 images) |
| `experiments_improvements_after_data_aug.ipynb` | Same 3 experiments (V2) — on the augmented balanced dataset |
| `evaluation_exp3_v2.ipynb` | Final model evaluation — Exp 3 V2, full metrics, visual predictions |

Each notebook represents a documented step in the scientific process: exploration → baseline → analysis → improvement → best model. The progression can be followed chronologically.

---

## 5. Training and Experiments

### Baseline

Trained on the original dataset (2,527 images, `trash` underrepresented at 137 images).

```
Test Accuracy : 84.50%
Trash Recall  : 70.4%   (model missed 30% of real trash items)
```

Global accuracy appeared acceptable, but per-class metrics revealed that the model was systematically avoiding the `trash` prediction. This is the key reason global accuracy alone is insufficient as an evaluation metric on imbalanced datasets.

### Learning curve interpretation

**Healthy training (observed):** Training and validation accuracy increase together, final gap approximately 5% (92.5% train / 87.5% val). No overfitting.

**Overfitting pattern (not observed):** Training accuracy continues rising while validation accuracy plateaus or decreases, gap exceeds 10% and grows.

**Underfitting (observed with EfficientNet):** Both curves remain low (below 60%) and stop improving after a few epochs.

A small dip in validation accuracy at epoch 7 (85.2% to 81.5%) is normal noise on a small dataset and does not indicate instability.

### Experiments V1 — `experiments_improvements.ipynb`

Three experiments were run on the original dataset to address the `trash` imbalance.

**Experiment 1 — Class Weights**

A weight inversely proportional to each class frequency is passed to `model.fit(class_weight=...)`. Errors on `trash` are penalized approximately 4x more heavily in the loss function. The model trains on the same data but learns to pay more attention to underrepresented classes.

Result: Trash recall improved significantly. Global accuracy gain limited because the underlying data scarcity remains.

**Experiment 2 — Fine-tuning**

The last 30 layers of MobileNetV2 are unfrozen and retrained at lr=1e-5 starting from the saved baseline weights. These higher-level layers encode complex patterns (shapes, surfaces) and can be adapted to differentiate waste-specific textures — distinguishing the reflective surface of metal from transparent glass, for example.

Result: Improved global accuracy. Trash recall limited by insufficient training examples.

**Experiment 3 — Combined (CW + Fine-tuning)**

Class weights and fine-tuning applied simultaneously. Best V1 result, still constrained by the original 137-image `trash` set.

### Offline data augmentation — `data_augmentation_trash.ipynb`

800 new `trash` images are physically generated and saved to disk (random flips, rotations, zooms, brightness variations). The `trash` class grows from 137 to 937 images. These are real files added to the dataset — not in-memory augmentation.

**Fundamental difference:** Class weights and fine-tuning compensate for data scarcity. Offline augmentation eliminates the problem at its source. The model genuinely learns to recognize `trash` because it has sufficient diverse examples, not because errors on that class are more costly.

### Experiments V2 — `experiments_improvements_after_data_aug.ipynb`

The same three experiments are rerun on the augmented dataset.

**Experiment 3 V2 (Data Augmentation + Class Weights + Fine-tuning)** produced the best overall result across all experiments and was selected as the final model.

---

## 6. Results

### Accuracy progression

| Version   | Technique                        | Test Accuracy | Trash Recall |
|-----------|----------------------------------|---------------|--------------|
| Baseline  | Transfer Learning (frozen)       | 84.5%         | 70.4%        |
| Exp 1 V1  | + Class Weights                  | ~86%          | ~85%         |
| Exp 2 V1  | + Fine-tuning                    | ~87%          | ~82%         |
| Exp 3 V1  | CW + Fine-tuning                 | ~87.5%        | ~87%         |
| Exp 3 V2  | Data Aug + CW + Fine-tuning      | **89.6%**     | **88.0%**    |

### Final model performance — Exp 3 V2

**Global metrics:**

| Metric        | Value   |
|---------------|---------|
| Test Accuracy | 89.62%  |
| Test Loss     | 0.3515  |
| Macro F1      | 0.893   |

**Per-class metrics:**

| Class     | Precision | Recall | F1    |
|-----------|-----------|--------|-------|
| cardboard | 0.907     | 0.925  | 0.916 |
| glass     | 0.841     | 0.881  | 0.860 |
| metal     | 0.863     | 0.932  | 0.896 |
| paper     | 0.924     | 0.924  | 0.924 |
| plastic   | 0.828     | 0.828  | 0.828 |
| trash     | 1.000     | 0.880  | 0.936 |

All six classes exceed 0.82 F1-score. Trash recall increased from 70.4% to 88.0% (+17.6 points). Trash precision of 1.000 indicates zero false positives — every prediction of `trash` by the model was correct.

A precision of 1.000 is not a sign of overfitting. Overfitting manifests in the learning curves (training accuracy rising while validation accuracy drops), not in individual class metrics on the test set. This value is confirmed by a recall of 0.880 — the model still misses 12% of real trash items, which is realistic and expected.

### Evaluation metrics used

**Accuracy** — global percentage of correct predictions. Insufficient alone on an imbalanced dataset.

**Loss (sparse_categorical_crossentropy)** — measures how far predicted probabilities are from the true class. Minimized during training.

**Precision** — of all images predicted as class X, how many actually belong to class X.

**Recall** — of all images that truly belong to class X, how many did the model correctly identify.

**F1-score** — harmonic mean of precision and recall. The primary comparison metric across experiments on an imbalanced dataset.

---

## 7. Application

A Streamlit web application allows real-time comparison of both models: `exp3_combined.keras` (before trash augmentation) and `v2_exp3_combined.keras` (final model).

**Features:**

- Upload any waste image (JPG, PNG, WEBP)
- Both models run simultaneously and return predictions
- Confidence score and probability distribution for all 6 classes per model
- Per-class metrics table (Precision, Recall, F1) displayed at all times
- Agreement / disagreement banner indicating whether both models predict the same class

**Running the application:**

```bash
pip install streamlit tensorflow pillow scikit-learn
streamlit run app/app.py
```

The application must be run from the project root. Both model files must be located in `./model/` relative to `app.py`.

---

## 8. Technical Notes

### Preprocessing — critical implementation detail

The `preprocess_input` normalization layer is baked into the model itself as a Lambda layer. During inference, the application must pass raw pixel values in the range [0, 255] directly to the model. Calling `preprocess_input` externally before passing the image to the model would result in double normalization, producing values outside the expected range and causing incorrect predictions. This was identified and corrected during development.

### Class name ordering

`image_dataset_from_directory` assigns class indices in alphabetical order based on subfolder names. The file `class_names.npy` stores the exact order used during training and must be loaded in the application to guarantee correct label mapping between model output indices and class names.

### Fine-tuning and catastrophic forgetting

When fine-tuning pre-trained layers, the learning rate must be significantly reduced (lr=1e-5, 100x smaller than the Adam default of 1e-3). A standard learning rate during fine-tuning overwrites the pre-trained ImageNet weights too aggressively, causing the model to lose its prior visual knowledge. The lower rate allows gradual adaptation.

### Data augmentation — online vs. offline

Online augmentation (Functional API with `training=True`) applies transformations in memory at each epoch and leaves files on disk unchanged. Offline augmentation (used in `data_augmentation_trash.ipynb`) generates and saves new image files permanently to the dataset folder. Offline augmentation was chosen for `trash` because the class was so underrepresented that online augmentation alone could not compensate for the genuine lack of visual diversity.

---

## 9. How to Run

### Requirements

```
tensorflow >= 2.12
numpy
matplotlib
seaborn
scikit-learn
streamlit
pillow
jupyter
```

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn streamlit pillow jupyter
```

### Training

```bash
# Open and run in order:
jupyter notebook notebooks/data_exploration.ipynb
jupyter notebook notebooks/training_MobileNet_CORRECTED.ipynb
jupyter notebook notebooks/evaluation_testing.ipynb
```

### Experiments

```bash
jupyter notebook notebooks/experiments_improvements.ipynb
jupyter notebook notebooks/evaluation_exp3.ipynb
jupyter notebook notebooks/data_augmentation_trash.ipynb
jupyter notebook notebooks/experiments_improvements_after_data_aug.ipynb
jupyter notebook notebooks/evaluation_after_data_aug.ipynb
```

### Application

```bash
streamlit run app/app.py
```

---

## Summary

| Item                | Details                                                        |
|---------------------|----------------------------------------------------------------|
| Architecture        | MobileNetV2 — CNN — Transfer Learning                         |
| Dataset             | 2,527 images (6 classes) — augmented to 3,327                 |
| Key challenge       | Class imbalance — `trash` at 5.4% of dataset                  |
| Solution            | Offline data augmentation + Class Weights + Fine-tuning        |
| Baseline accuracy   | 84.5%                                                          |
| Final accuracy      | 89.6%                                                          |
| Trash recall        | 70.4% → 88.0% (+17.6 points)                                  |
| Macro F1            | 0.893                                                          |
| Deployment          | Streamlit app — dual model real-time comparison                |