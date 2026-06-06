"""CNN preprocessing helpers for image tensors and ResNet50 inputs"""
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import tensorflow as tf

from src.config import CNN_IMAGE_SIZE
from src.data import ImageRecord


# 96x96 keeps training practical for this course project while preserving enough
# bin shape and fill-level detail for this dataset.
IMAGENET_BGR_MEAN = (103.939, 116.779, 123.68)


def image_to_cnn_tensor(
    img: np.ndarray,
    image_size: tuple[int, int] = CNN_IMAGE_SIZE,
) -> np.ndarray:
    """Convert a BGR OpenCV image to one normalized RGB CNN input tensor

    Args:
        img: OpenCV image in BGR channel order
        image_size: Target width and height for CNN input

    Returns:
        Float32 RGB image tensor scaled to the 0-1 range
    """

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, image_size, interpolation=cv2.INTER_AREA)
    return resized.astype("float32") / 255.0


def records_to_tensors(
    records: Sequence[ImageRecord],
    image_size: tuple[int, int] = CNN_IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Load image records as CNN tensors plus integer labels

    Args:
        records: Annotated image records to load from disk
        image_size: Target width and height for CNN input

    Returns:
        Image tensor array and integer label array

    Raises:
        ValueError: If an image cannot be read from disk
    """

    images = []
    labels = []
    for record in records:
        img = cv2.imread(str(record.image_path))
        if img is None:
            raise ValueError(f"Image could not be read: {record.image_path}")
        images.append(image_to_cnn_tensor(img, image_size=image_size))
        labels.append(record.label)
    return np.asarray(images), np.asarray(labels)


def apply_resnet50_preprocessing(inputs: Any) -> Any:
    """Convert RGB tensors to the BGR ImageNet format expected by ResNet50

    Args:
        inputs: RGB input tensor scaled to the 0-1 range

    Returns:
        Tensor normalized with ImageNet BGR channel means
    """

    x = tf.keras.layers.Rescaling(scale=255.0)(inputs)
    x = tf.keras.layers.Concatenate(axis=-1)(
        [
            x[..., 2:3],  # Blue channel
            x[..., 1:2],  # Green channel
            x[..., 0:1],  # Red channel
        ]
    )
    return tf.keras.layers.Normalization(
        axis=-1,
        mean=IMAGENET_BGR_MEAN,
        variance=(1.0, 1.0, 1.0),
        name="resnet50_preprocessing",
    )(x)