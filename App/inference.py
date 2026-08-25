import os
import numpy as np
import tensorflow as tf

from App.preprocessing import preprocess_mri


# ============================================================
# 1. Model Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASSIFICATION_MODEL_PATH = os.path.join(
    BASE_DIR,
    "Models",
    "brain_tumor_classifier.keras"
)

SEGMENTATION_MODEL_PATH = os.path.join(
    BASE_DIR,
    "Models",
    "brain_tumor_segmenter.keras"
)


# ============================================================
# 2. Load Models
# ============================================================

classification_model = tf.keras.models.load_model(
    CLASSIFICATION_MODEL_PATH,
    compile=False
)

segmentation_model = tf.keras.models.load_model(
    SEGMENTATION_MODEL_PATH,
    compile=False
)


# ============================================================
# 3. Class Names
# ============================================================

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]


# ============================================================
# 4. Segmentation Function
# ============================================================

def run_segmentation(processed_image):

    # -----------------------------
    # 1. Segmentation prediction
    # -----------------------------
    segmentation_probability = segmentation_model.predict(
        processed_image,
        verbose=0
    )[0, :, :, 0]

    # -----------------------------
    # 2. Convert probability map
    #    to binary mask
    # -----------------------------
    segmentation_mask = (
        segmentation_probability >= 0.5
    ).astype(np.uint8)

    return {
        "probability": segmentation_probability,
        "mask": segmentation_mask
    }


# ============================================================
# 5. End-to-End Inference Function
# ============================================================

def run_inference(image_path):

    # -----------------------------
    # 1. Preprocessing
    # -----------------------------
    original_image, processed_image = preprocess_mri(
        image_path
    )

    # -----------------------------
    # 2. Classification
    # -----------------------------
    classification_prediction = classification_model.predict(
        processed_image,
        verbose=0
    )

    predicted_class_index = np.argmax(
        classification_prediction[0]
    )

    predicted_class = CLASS_NAMES[
        predicted_class_index
    ]

    confidence = float(
        np.max(classification_prediction[0])
    )

    # -----------------------------
    # 3. Tumor / No Tumor decision
    # -----------------------------
    if predicted_class == "no_tumor":

        return {
            "image": original_image,
            "prediction": predicted_class,
            "confidence": confidence,
            "segmentation": None,
            "segmentation_probability": None
        }

    # -----------------------------
    # 4. Segmentation
    # -----------------------------
    segmentation_result = run_segmentation(
        processed_image
    )

    return {
        "image": original_image,
        "prediction": predicted_class,
        "confidence": confidence,
        "segmentation": segmentation_result["mask"],
        "segmentation_probability": segmentation_result["probability"]
    }