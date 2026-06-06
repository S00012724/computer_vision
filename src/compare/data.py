"""
Data and evaluation functions for the classical comparison model

The comparison model uses HOG features plus a Linear SVM. It gives the report a
handcrafted-feature baseline without adding another deep model
"""
import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import TEST_RATIO
from src.data import (
    load_annotations,
    resolve_data_dir,
    split_records,
)

# The HOG baseline uses this fixed input size to keep feature extraction fast
# and give every image the same descriptor length for the Linear SVM
HOG_IMAGE_SIZE = (64, 64)


def extract_hog_features(
    img: np.ndarray,
    win_size: tuple[int, int] = HOG_IMAGE_SIZE,
    block_size: tuple[int, int] = (16, 16),
    block_stride: tuple[int, int] = (8, 8),
    cell_size: tuple[int, int] = (8, 8),
    nbins: int = 9,
) -> np.ndarray:
    """Extract OpenCV HOG features with a fixed descriptor layout

    Parameters use the common OpenCV HOG layout from the course topic: 8x8
    cells, 16x16 blocks, 8-pixel block stride, and 9 orientation bins. The image
    is converted to grayscale because HOG measures intensity gradients

    Args:
        img: OpenCV image in BGR channel order
        win_size: Descriptor window size in pixels
        block_size: HOG block size in pixels
        block_stride: HOG block stride in pixels
        cell_size: HOG cell size in pixels
        nbins: Number of orientation histogram bins

    Returns:
        One-dimensional HOG feature vector
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if (gray.shape[1], gray.shape[0]) != win_size:
        gray = cv2.resize(gray, win_size)

    descriptor = cv2.HOGDescriptor(
        win_size,
        block_size,
        block_stride,
        cell_size,
        nbins,
    )
    return descriptor.compute(gray).ravel()


def _records_to_hog_features(records, img_size=HOG_IMAGE_SIZE):
    """Convert image records into HOG feature rows and label arrays

    Args:
        records: Annotated image records to load from disk
        img_size: Target width and height for HOG extraction

    Returns:
        HOG feature matrix and integer label array

    Raises:
        ValueError: If an image cannot be read from disk
    """

    features = []
    labels = []
    for record in records:
        img = cv2.imread(str(record.image_path))
        if img is None:
            raise ValueError(f"Image could not be read: {record.image_path}")

        features.append(extract_hog_features(img, win_size=img_size))
        labels.append(record.label)

    return np.asarray(features), np.asarray(labels)


def get_comparison_train_test_data(
    data_dir=None,
    img_size=HOG_IMAGE_SIZE,
    test_ratio=TEST_RATIO,
):
    """Return HOG train and test arrays for the Linear SVM baseline

    Args:
        data_dir: Dataset directory or None to use the configured dataset
        img_size: Target width and height for HOG extraction
        test_ratio: Fraction of each class to place in the test split

    Returns:
        Train arrays, test arrays, class names, split counts, and data path
    """

    data_path = resolve_data_dir(data_dir)
    records, class_names, counts = load_annotations(data_path)
    train_records, test_records, split_info = split_records(
        records,
        test_ratio=test_ratio,
    )

    X_train, y_train = _records_to_hog_features(
        train_records,
        img_size=img_size,
    )
    X_test, y_test = _records_to_hog_features(
        test_records,
        img_size=img_size,
    )
    split_counts = {
        "all": counts,
        **split_info,
    }
    return (
        X_train,
        y_train,
        X_test,
        y_test,
        class_names,
        split_counts,
        str(data_path),
    )


def build_comparison_model():
    """Build the HOG and Linear SVM comparison model used in the report

    StandardScaler normalizes the HOG descriptor columns before SVM training
    class_weight="balanced" helps the smaller needs_collection class count more
    during training. random_state=42 keeps the result reproducible

    Returns:
        Model name, feature-set label, and scikit-learn estimator pipeline
    """

    return (
        "OpenCV HOG 64x64 + Linear SVM",
        "OpenCV HOG 64x64",
        Pipeline([
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(
                class_weight="balanced",
                max_iter=20000,
                random_state=42,
                dual="auto",
            )),
        ]),
    )


def evaluate_candidate(name, estimator, X_train, y_train, X_test, y_test, labels):
    """Train and evaluate one model on the shared held-out split

    The metrics mirror the CNN evaluation so the two models can be compared in
    one table: accuracy, macro precision, macro recall, macro F1, and confusion
    matrix

    Args:
        name: Display name for the model
        estimator: scikit-learn estimator with fit and predict methods
        X_train: Training feature matrix
        y_train: Training labels
        X_test: Test feature matrix
        y_test: Test labels
        labels: Ordered class indexes used for metric labels

    Returns:
        Model name, classification metrics, and confusion matrix
    """

    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_precision": precision_score(
            y_test,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_test,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_test,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
    }
