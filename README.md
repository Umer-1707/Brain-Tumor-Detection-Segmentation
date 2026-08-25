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

## Result

The final pipeline successfully performs MRI preprocessing, tumor classification, and tumor segmentation through a single application.

The resulting Streamlit interface displays the prediction, confidence score, predicted tumor mask, and an optional overlay on the original MRI.

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
git clone <repository-url>
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