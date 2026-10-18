# 🦅 Artimis Bird Classifier — CNN Assignment

> With a custom 20-class Dataset made a deep-learning pipeline for **Indian bird species classification** using PyTorch. Compares a custom-built CNN (from scratch) against a pretrained ResNet-18 transfer-learning baseline.


## 🧠 Project Overview

This project trains a bird image classifier across **20 Indian bird species** using two approaches:

| Approach | Description | Test Accuracy |
|----------|-------------|---------------|
| **Custom CNN** | Built from scratch, 10 epochs | ~40% |
| **Pretrained ResNet-18** | Transfer learning (ImageNet weights) | **~93%** |

The notebook is fully self-contained and walks through:
- Environment verification (GPU check)
- Reproducibility seeding
- Dataset splitting (stratified 80/10/10)
- Data augmentation & DataLoaders
- Sanity checks
- Two training experiments with full evaluation

---

## 📂 Dataset

- **Total Images**: 18,055
- **Number of Classes**: 20 Indian bird species (e.g., Common Moorhen, Brahminy Kite, Black-winged Kite, Common Crane, Brahminy Starling...)
- **Split Strategy**: Stratified split to preserve class balance
  - Train: 14,444 images (80%)
  - Validation: 1,805 images (10%)
  - Test: 1,806 images (10%)

---

## 🖥️ Environment & Requirements

#### Remove GPU code and run on Google volab and kaggle for claoud base GPU training.
---

### Tested On

| Component | Version |
|-----------|---------|
| Python | 3.12.10 |
| PyTorch | 2.14.1+cu132 |
| Torchvision | 0.29.1+cu132 |
| CUDA | 13.2 |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU (8.55 GB) |

### Python Dependencies

```
torch
torchvision
numpy
pandas
matplotlib
seaborn
Pillow
scikit-learn
```

---


## 📊 Experiments & Results

### Experiment 1 — Custom CNN (from Scratch)

- **Architecture**: Custom CNN, ~531K parameters
- **Optimizer**: Adam
- **Epochs**: 10
- **Loss**: Cross-Entropy with class weights

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|---------|---------|
| 1     | 2.7656    | 13.33%    | 2.5222  | 18.12%  |
| 5     | 2.3035    | 25.78%    | 2.1415  | 29.14%  |
| 10    | 1.9986    | 35.98%    | 1.8425  | **40.72%** |

> Training time: ~47 minutes (RTX 5050 GPU)

---

### Experiment 2 — Pretrained ResNet-18 (Transfer Learning)

- **Base Model**: ResNet-18 (ImageNet pretrained)
- **Fine-tuned**: Final FC layer replaced for 20 classes

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **93.02%** |
| Macro F1 | 91.23% |
| Weighted F1 | 92.93% |

> Transfer learning gives a **2.3x improvement** in accuracy over training from scratch.

---