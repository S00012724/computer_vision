import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import CNN_IMAGE_SIZE, CNN_MODEL_PATH, DATA_DIR
from src.data import load_annotations, resolve_data_dir, split_records
from src.models import (
    _evaluate_predictions,
)
from src.preprocessing import (
    apply_resnet50_preprocessing,
    records_to_tensors,
)


TRAIN_EPOCHS = 5
TRAIN_BATCH_SIZE = 32
TRAIN_TRANSFER_BASE_NAME = "ResNet50"


def build_cnn(
    input_shape=(96, 96, 3),
    num_classes=2,
    weights="imagenet",
    base_trainable=False,
):
    """Build the ResNet50 transfer-learning classifier

    `include_top=False` removes the ImageNet classifier so the project can add
    its own two-class head. The base starts frozen (`base_trainable=False`)
    because the dataset is small. Dropout 0.3 reduces overfitting before the
    final softmax scores

    Args:
        input_shape: Image tensor shape expected by the model
        num_classes: Number of output classes
        weights: ResNet50 weight source passed to Keras
        base_trainable: Whether to fine-tune the ResNet50 base

    Returns:
        Compiled Keras classification model
    """

    base_model = tf.keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )
    base_model.trainable = base_trainable

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = apply_resnet50_preprocessing(inputs)
    if base_trainable:
        x = base_model(x)
    else:
        x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="waste_bin_resnet50",
    )
    # Adam updates the weights, sparse categorical cross-entropy matches the
    # integer labels, and accuracy is reported as an easy training metric
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate_cnn(
    image_size=CNN_IMAGE_SIZE,
):
    """Train the CNN, save it, and return held-out test metrics

    Defaults match the reported run: 5 epochs, batch size 32, 96x96 images, and
    the shared stratified hold-out split from src.data

    Args:
        image_size: Width and height used to resize images before training

    Returns:
        Training metadata and held-out test metrics
    """

    tf.keras.utils.set_random_seed(42)

    data_path = resolve_data_dir(DATA_DIR)
    records, class_names, counts = load_annotations(data_path)
    train_records, test_records, split_counts = split_records(records)

    X_train, y_train = records_to_tensors(train_records, image_size=image_size)
    X_test, y_test = records_to_tensors(test_records, image_size=image_size)

    model = build_cnn(
        input_shape=(image_size[1], image_size[0], 3),
        num_classes=len(class_names),
    )

    start_time = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=TRAIN_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        verbose=2,
    )
    training_seconds = time.perf_counter() - start_time

    model_path = Path(CNN_MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    probabilities = model.predict(X_test, verbose=0)
    y_pred = probabilities.argmax(axis=1)
    result = _evaluate_predictions(y_test, y_pred, class_names)
    return {
        **result,
        "data_path": str(data_path),
        "class_names": class_names,
        "split_counts": {"all": counts, **split_counts},
        "epochs": TRAIN_EPOCHS,
        "batch_size": TRAIN_BATCH_SIZE,
        "training_seconds": training_seconds,
        "model_path": str(model_path),
    }


def print_cnn_results(result):
    """Print training and test results in the report format

    Args:
        result: Metrics and metadata returned by train_and_evaluate_cnn
    """

    print(f"--- {TRAIN_TRANSFER_BASE_NAME} Transfer-Learning CNN ---")
    print(f"Dataset: {result['data_path']}")
    print(f"Saved model: {result['model_path']}")
    print(f"Test strategy: {result['split_counts']['test_strategy']}")
    print(f"Train counts: {result['split_counts']['train']}")
    print(f"Test counts: {result['split_counts']['test']}")
    print(f"Epochs: {result['epochs']}")
    print(f"Batch size: {result['batch_size']}")
    print(f"Training time: {result['training_seconds']:.2f} seconds")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Macro Precision: {result['macro_precision']:.4f}")
    print(f"Macro Recall: {result['macro_recall']:.4f}")
    print(f"Macro F1: {result['macro_f1']:.4f}")
    print("Confusion matrix rows/columns:", result["class_names"])
    print(np.asarray(result["confusion_matrix"]))


def parse_args():
    """Parse command-line arguments for CNN training

    Returns:
        Parsed argparse namespace
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train the Waste-bin fill-level model with the fixed report "
            "configuration."
        )
    )
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    print_cnn_results(train_and_evaluate_cnn())
