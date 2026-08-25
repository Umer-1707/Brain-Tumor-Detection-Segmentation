import cv2
import numpy as np


def preprocess_mri(image_path):
    """
    Preprocess an MRI image for model inference.

    Returns:
        original_image: Grayscale MRI before resizing
        processed_image: Preprocessed image ready for the model
    """

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to model input size
    resized = cv2.resize(
        gray,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    # Normalize pixel values
    normalized = resized.astype(np.float32) / 255.0

    # Add channel dimension: (224, 224) → (224, 224, 1)
    processed = np.expand_dims(normalized, axis=-1)

    # Add batch dimension: (224, 224, 1) → (1, 224, 224, 1)
    processed = np.expand_dims(processed, axis=0)

    return gray, processed