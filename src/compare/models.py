"""
Evaluate the HOG + Linear SVM comparison model
"""
import argparse

import numpy as np

from src.config import DATA_DIR
from src.compare.data import (
    build_comparison_model,
    evaluate_candidate,
    get_comparison_train_test_data,
)


def compare_models(data_dir=None):
    """Evaluate the configured comparison model on the dataset

    Args:
        data_dir: Dataset directory or None to use the configured dataset

    Returns:
        Results list, class names, split counts, and dataset path
    """

    X_train, y_train, X_test, y_test, class_names, split_counts, data_path = (
        get_comparison_train_test_data(data_dir=data_dir)
    )
    labels = list(range(len(class_names)))
    name, feature_set, estimator = build_comparison_model()
    result = evaluate_candidate(
        name,
        estimator,
        X_train,
        y_train,
        X_test,
        y_test,
        labels=labels,
    )
    result["feature_set"] = feature_set
    return [result], class_names, split_counts, str(data_path)


def print_results(results, class_names, split_counts, data_path):
    """Print comparison metrics in report-table format

    Args:
        results: Evaluation result dictionaries from compare_models
        class_names: Ordered class labels used by the dataset
        split_counts: Train and test count metadata
        data_path: Dataset path shown in console output
    """

    print("--- HOG + Linear SVM Comparison Model ---")
    print(f"Dataset: {data_path}")
    print(f"Test strategy: {split_counts['test_strategy']}")
    print(f"Train counts: {split_counts['train']}")
    print(f"Test counts: {split_counts['test']}")
    print()
    print("| Model | Features | Accuracy | Macro Precision | Macro Recall | Macro F1 |")
    print("|---|---|---:|---:|---:|---:|")
    for result in results:
        print(
            f"| {result['model']} | {result['feature_set']} | "
            f"{result['accuracy']:.4f} | "
            f"{result['macro_precision']:.4f} | "
            f"{result['macro_recall']:.4f} | "
            f"{result['macro_f1']:.4f} |"
        )

    print()
    print("Confusion matrix rows/columns:", class_names)
    print(np.asarray(results[0]["confusion_matrix"]))


def parse_args():
    """Parse command-line arguments for comparison evaluation

    Returns:
        Parsed argparse namespace with dataset path
    """

    parser = argparse.ArgumentParser(
        description="Evaluate the HOG + Linear SVM comparison model."
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    comparison, names, counts, path = compare_models(data_dir=args.data_dir)
    print_results(comparison, names, counts, path)
