# Brain Tumor Detection & Segmentation

A deep learning project for detecting and segmenting brain tumors from MRI images. The project started as separate notebook-based workflows for classification and segmentation and was eventually combined into a single end-to-end Streamlit application.

## Overview

The final system combines two trained models into one inference pipeline:

1. A **CNN-based classification model** that classifies an MRI slice as **glioma, meningioma, pituitary tumor, or no tumor**.
2. A **Residual U-Net segmentation model** that predicts the tumor region when a tumor is detected.

The models are integrated into a Streamlit web application where users can upload an MRI image, view the predicted class and confidence score, inspect the predicted tumor mask, overlay it on the original MRI, and download the mask.

## What the Application Does

1. Launch the application with `python main.py`.
2. Streamlit opens the web interface.
3. Upload an MRI image.
4. The image is temporarily saved and preprocessed.
5. The classification model predicts one of four classes.
6. The predicted class and confidence score are displayed.
7. If a tumor is detected, the segmentation model predicts its region.
8. The predicted mask is displayed alongside the original MRI.
9. The mask can be overlaid on the original scan.
10. The predicted mask can be downloaded as a PNG file.

The UI uses a dark teal/blue visual theme with custom typography, result cards, status badges, a confidence bar, and a scan animation during inference.

## Model Development

The models were developed and evaluated across three Jupyter notebooks.

### Classification — `Brain_Tumor_Classification.ipynb`

- Extracted and inspected the classification dataset obtained through the Kaggle API.
- Organized the dataset and visualized random samples from each class.
- Preprocessed the images and created a dataset index.
- Used stratified splitting with a **70/15/15 train/validation/test split**.
- Verified class distributions and checked for data leakage.
- Built a TensorFlow input pipeline for loading and normalizing images.
- Added training-data augmentation and verified the final data pipeline.
- Checked validation and test datasets and handled class imbalance using class weights.
- Designed and compiled a CNN architecture.
- Configured training callbacks and trained the model.
- Visualized training and validation performance.
- Evaluated the model using predictions, a classification report, confusion matrix, and tumor-vs-no-tumor analysis.
- Saved the final classification model.

### Segmentation — `Brain_Tumor_Segmentation.ipynb`

- Extracted and inspected the segmentation dataset.
- Verified MRI and tumor-mask pairs.
- Applied separate preprocessing techniques to MRI images and masks.
- Visualized the preprocessed images and masks.
- Split the dataset and checked for data leakage.
- Built and trained an initial U-Net model.
- Used **Dice coefficient** for measuring tumor overlap, **Dice loss** for optimization, a combination of **binary cross-entropy and Dice loss**, and **IoU** for evaluation.
- Evaluated the initial U-Net and found its results unsatisfactory.
- Designed a **Residual U-Net** to improve segmentation performance.
- Compiled and trained the Residual U-Net with training callbacks.
- Evaluated the model on the test set.
- Compared ground-truth masks with predicted masks and visualized the results.
- Saved the final Residual U-Net model used by the application.

The segmentation model uses a custom loss function; its registration is handled in `inference.py` before the model is loaded.

### End-to-End Evaluation — `Final_Evaluation.ipynb`

This notebook was used to verify that both trained models worked together as a complete inference pipeline before building the application.

- Loaded both trained models.
- Loaded test MRI images.
- Created preprocessing and inference functions.
- Ran classification and segmentation together.
- Visualized the final predicted tumor masks.

## Results

Both models were evaluated on held-out test sets (never seen during training or validation) to confirm they generalize to unseen data, not just perform well on training data.

---

### Classification — 4-Class CNN

Trained on 10,560 preprocessed MRI images (grayscale, 224×224), stratified 70/15/15 split (7,392 train / 1,584 val / 1,584 test). Class weights were applied to offset the natural imbalance across classes (`no_tumor` was the minority class at 1,757 images vs. 3,754 for `glioma`). Train/val/test splits were verified to have zero overlap.

**Test set performance:**

| Metric | Score |
|---|---|
| Test Accuracy | **88.38%** |
| Test Loss | 0.3022 |
| Model size | 617K params (2.35 MB) |

**Per-class metrics (test set, n = 1,584):**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Glioma | 0.9463 | 0.8757 | 0.9096 | 563 |
| Meningioma | 0.7597 | 0.7812 | 0.7703 | 352 |
| No Tumor | 0.9163 | 0.9163 | 0.9163 | 263 |
| Pituitary | 0.8927 | 0.9631 | 0.9265 | 406 |
| **Macro avg** | 0.8787 | 0.8841 | 0.8807 | 1,584 |
| **Weighted avg** | 0.8861 | 0.8838 | 0.8841 | 1,584 |

**Clinically-relevant check — missed tumor rate:** of 1,321 actual tumor-positive test images, only **22 were misclassified as `no_tumor`** — a **1.67% false-negative rate**. This matters more than raw accuracy for a screening system, since missing a real tumor is a far costlier error than confusing one tumor type for another.

Meningioma is the weakest class (F1 0.77) and the main driver of overall classification error — a known limitation and a target for future improvement (more meningioma samples, finer-grained augmentation).

---

### Segmentation — Residual U-Net

Trained on 3,063 paired MRI/mask images (grayscale, 224×224, binary masks), split 2,144 / 459 / 460 (train/val/test), with confirmed zero overlap between splits. Loss: combined Binary Cross-Entropy + Dice loss. Optimizer used a `ReduceLROnPlateau` schedule with early stopping.

**Baseline U-Net vs. Residual U-Net:**

| Model | Best Val Dice | Best Val IoU | Verdict |
|---|---|---|---|
| Baseline U-Net | ~0.668 | ~0.504 | Insufficient — replaced |
| **Residual U-Net** | **0.8247** | **0.7047** | Adopted |

Switching from a standard U-Net to a Residual U-Net (residual blocks with skip connections in encoder and decoder) lifted validation Dice from ~0.67 to ~0.82 and IoU from ~0.50 to ~0.70 — the reason the baseline was dropped in favor of this architecture.

**Final Residual U-Net — held-out test set performance:**

| Metric | Score |
|---|---|
| Dice coefficient | **0.8228** |
| IoU score | **0.7041** |
| Test loss (BCE + Dice) | 0.2092 |
| Pixel-level Precision | 0.8589 |
| Pixel-level Recall | 0.7966 |
| Model size | 8.1M params (30.98 MB) |

Test Dice/IoU (0.8228 / 0.7041) closely track the restored checkpoint's validation numbers (0.8245 / 0.7039) — a good sign the model generalizes consistently rather than overfitting to the validation split specifically. Precision (0.86) trailing above Recall (0.80) suggests the model is slightly conservative — it occasionally under-segments (misses some tumor pixels) rather than over-segmenting healthy tissue as tumor.

---

### End-to-End Pipeline

Verified by chaining both trained models on held-out sample MRIs: a glioma test scan was classified with **91.58% confidence**, correctly triggering the segmentation stage, which produced a tumor mask closely aligned with the ground-truth region — confirming the two independently trained models work correctly together as a single inference pipeline before being wrapped in the Streamlit app.

---

> **Metric notes:** Precision/Recall/F1 are standard classification metrics. Dice coefficient and IoU (Intersection over Union) both measure overlap between predicted and ground-truth tumor regions at the pixel level — Dice weights overlap more heavily, IoU is the stricter, more conservative measure. Both are standard for evaluating medical image segmentation.

## Application Structure

```text
App/
├── app.py
├── inference.py
└── preprocessing.py
```

- **`preprocessing.py`** — contains the final preprocessing function used to prepare an uploaded MRI image for inference.
- **`inference.py`** — contains the inference pipeline. It preprocesses the image, loads and runs both trained models, returns the classification result and confidence, and runs segmentation when appropriate.
- **`app.py`** — contains the Streamlit user interface, including image upload, scanning visualization, classification results, confidence display, segmentation mask, mask overlay, mask download, and disclaimer.

`main.py` is the main entry point of the project and launches the Streamlit application.

## Project Structure

```
Brain Tumor Detection & Segmentation/
│
├── App/
│   ├── app.py
│   ├── inference.py
│   └── preprocessing.py
│
├── Models/
│   ├── brain_tumor_classifier.keras
│   └── brain_tumor_segmenter.keras
│
├── Notebooks/
│   ├── Brain_Tumor_Classification.ipynb
│   ├── Brain_Tumor_Segmentation.ipynb
│   └── Final_Evaluation.ipynb
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

The training datasets, Kaggle credentials, virtual environment, Python cache files, and test images are not included in the repository.

## Installation

Clone the repository and move into the project directory:

```bash
git clone https://github.com/Umer-1707/Brain-Tumor-Detection-Segmentation.git
cd "Brain Tumor Detection & Segmentation"
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Application

The intended entry point is `main.py`:

```bash
python main.py
```

This launches the Streamlit web application locally.

You do not need to run Streamlit directly through the command line. `main.py` handles the application launch.

## Models

| Model | File | Purpose |
|---|---|---|
| Classification | `Models/brain_tumor_classifier.keras` | CNN model that classifies MRI slices into glioma, meningioma, pituitary, or no tumor |
| Segmentation | `Models/brain_tumor_segmenter.keras` | Final Residual U-Net model that predicts the tumor region |

## Datasets

Two separate datasets were used during development.

### Classification Dataset

The classification model was trained using the [Brain Tumor Dataset](https://www.kaggle.com/datasets/ishans24/brain-tumor-dataset) obtained through the Kaggle API.

The dataset is not included in this repository.

### Segmentation Dataset

A separate dataset containing paired MRI images and tumor masks was used to train the segmentation model.

The original source of this dataset is not currently available, so no source link is provided here. The dataset is also not included in the repository.

## Technologies

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pillow
- Streamlit
- Jupyter Notebook
- Kaggle API

## Future Improvements

- Improve classification and segmentation performance.
- Train with larger and more diverse MRI datasets to improve generalization.
- Reduce the classifier's tendency to default to `no_tumor` on MRI images that differ from the training data.
- Add quantitative segmentation metrics and support for additional MRI modalities.
- Optimize inference and explore deployment as a hosted application.

## Disclaimer

This project is intended for research and educational purposes only. It is **not a medical diagnostic system** and should not be used as a substitute for evaluation by a qualified medical professional.

The classification model was trained on a single dataset and has shown reduced reliability on MRI images that differ significantly from the training data, including images from different sources or scanners. Predictions should therefore not be interpreted as medical diagnoses.

## Author

**Muhammad Umer**
