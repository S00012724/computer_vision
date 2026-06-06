"""
Shared evaluation helpers for the waste-bin fill-level models

The saved CNN is evaluated with metrics close to the assignment: accuracy,
macro precision, macro recall, macro F1, and a confusion matrix
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import tensorflow as tf

from src.config import CNN_IMAGE_SIZE, CNN_MODEL_PATH
from src.data import (
    load_annotations,
    resolve_data_dir,
    split_records,
)
from src.postprocessing import (
    LOW_PRIORITY_RISK_THRESHOLD,
    URGENT_RISK_THRESHOLD,
    postprocess_prediction,
)
from src.preprocessing import records_to_tensors


def _evaluate_predictions(y_test, y_pred, class_names):
    """Compute the classification metrics required by the assignment

    Args:
        y_test: Ground-truth class indexes for the test split
        y_pred: Predicted class indexes for the test split
        class_names: Ordered class labels used to define metric labels

    Returns:
        Accuracy, macro metrics, and confusion matrix
    """

    labels = list(range(len(class_names)))
    return {
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


def _cnn_class_score_rows(probabilities, class_names):
    """Format softmax rows as class-name dictionaries for post-processing

    Args:
        probabilities: Model probability rows for each image
        class_names: Ordered class labels matching each probability column

    Returns:
        List of per-image class score mappings
    """

    return [
        {
            class_name: float(probability)
            for class_name, probability in zip(class_names, row)
        }
        for row in probabilities
    ]


def evaluate_cnn_model(
    model_path=CNN_MODEL_PATH,
    data_dir=None,
    image_size=CNN_IMAGE_SIZE,
):
    """Load the saved CNN and evaluate it on the held-out test split

    The post-processing section measures what happens after uncertain
    needs_collection risk scores move to manual review. It does not retrain the
    model and applies fixed thresholds to the CNN scores

    Args:
        model_path: Path to the saved Keras model
        data_dir: Dataset directory or None to use the configured dataset
        image_size: Width and height used to resize images before inference

    Returns:
        Metrics, split details, and post-processing outputs
    """

    data_path = resolve_data_dir(data_dir)
    model_path = Path(model_path)
    print(f"Loading CNN model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    records, class_names, counts = load_annotations(data_path)
    _, test_records, split_counts = split_records(records)
    X_test, y_test = records_to_tensors(test_records, image_size=image_size)

    probabilities = model.predict(X_test, verbose=0)
    y_pred = probabilities.argmax(axis=1)
    result = _evaluate_predictions(y_test, y_pred, class_names)
    confidence_scores = probabilities[np.arange(len(y_pred)), y_pred]
    class_score_rows = _cnn_class_score_rows(probabilities, class_names)
    labels = list(range(len(class_names)))

    print("\n--- CNN Evaluation Results ---")
    print(f"Dataset: {data_path}")
    print(f"Model: {model_path}")
    print(f"Test strategy: {split_counts['test_strategy']}")
    print(f"Test images: {sum(split_counts['test'].values())}")
    print(f"Test counts: {split_counts['test']}")
    print(f"Accuracy:  {result['accuracy']:.4f}")
    print(f"Precision: {result['macro_precision']:.4f} (Macro)")
    print(f"Recall:    {result['macro_recall']:.4f} (Macro)")
    print(f"F1-Score:  {result['macro_f1']:.4f} (Macro)")
    print("\nConfusion Matrix:")
    print(result["confusion_matrix"])
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    ))

    print("\n--- Post-Processing Risk-Score Policy ---")
    postprocessed = [
        postprocess_prediction(
            class_names[prediction],
            confidence=float(confidence),
            class_scores=class_scores,
        )
        for prediction, confidence, class_scores in zip(y_pred, confidence_scores, class_score_rows)
    ]
    priorities = [row["collection_priority"] for row in postprocessed]
    review_count = priorities.count("review")
    auto_mask = np.array(priorities) != "review"
    review_mask = ~auto_mask
    auto_accuracy = accuracy_score(y_test[auto_mask], y_pred[auto_mask]) if auto_mask.any() else 0.0
    auto_confusion = (
        confusion_matrix(y_test[auto_mask], y_pred[auto_mask], labels=labels)
        if auto_mask.any()
        else np.zeros((len(labels), len(labels)), dtype=int)
    )
    needs_collection_index = class_names.index("needs_collection")
    has_space_index = class_names.index("has_space")
    auto_false_low = int(auto_confusion[needs_collection_index, has_space_index])
    auto_recalls = (
        recall_score(
            y_test[auto_mask],
            y_pred[auto_mask],
            labels=labels,
            average=None,
            zero_division=0,
        )
        if auto_mask.any()
        else np.zeros(len(labels), dtype=float)
    )
    auto_needs_collection_recall = float(auto_recalls[needs_collection_index])
    review_errors = int(np.sum(y_test[review_mask] != y_pred[review_mask]))
    risk_levels = [row["risk_level"] for row in postprocessed]

    print(
        "Collection priorities generated: "
        f"{priorities.count('urgent')} urgent, "
        f"{priorities.count('low')} low, "
        f"{review_count} review"
    )
    print(
        f"Risk policy: urgent when needs_collection score >= "
        f"{URGENT_RISK_THRESHOLD:.2f}; low when needs_collection score <= "
        f"{LOW_PRIORITY_RISK_THRESHOLD:.2f}; otherwise review; "
        f"reviewed {review_count}/{len(y_pred)} predictions "
        f"({review_count / len(y_pred):.1%})"
    )
    print(
        "Risk levels: "
        f"{risk_levels.count('high')} high, "
        f"{risk_levels.count('medium')} medium, "
        f"{risk_levels.count('low')} low, "
        f"{risk_levels.count('unknown')} unknown"
    )
    print(f"Accuracy on non-review predictions: {auto_accuracy:.4f}")
    print("Automatic confusion matrix rows/columns:", class_names)
    print(auto_confusion)
    print(f"Full bins still missed automatically: {auto_false_low}")
    print(f"Automatic full-bin recall: {auto_needs_collection_recall:.4f}")
    print(f"Errors captured in review queue: {review_errors}")

    return {
        **result,
        "data_path": str(data_path),
        "class_names": class_names,
        "split_counts": {"all": counts, **split_counts},
        "model_path": str(model_path),
        "priorities": priorities,
        "review_count": review_count,
        "auto_accuracy": auto_accuracy,
        "auto_confusion_matrix": auto_confusion,
        "auto_false_low": auto_false_low,
        "auto_needs_collection_recall": auto_needs_collection_recall,
        "review_errors": review_errors,
        "postprocessing": postprocessed,
    }
