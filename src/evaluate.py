import argparse
import os

from src.config import CNN_MODEL_PATH, DATA_DIR
from src.models import evaluate_cnn_model


def parse_args():
    """Parse command-line arguments for CNN evaluation

    Returns:
        Parsed argparse namespace with dataset and model paths
    """

    parser = argparse.ArgumentParser(
        description="Evaluate the model on its held-out test split."
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--model", default=str(CNN_MODEL_PATH))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if os.path.exists(args.model):
        evaluate_cnn_model(model_path=args.model, data_dir=args.data_dir)
    else:
        print(f"Model file not found: {args.model}. Please run train.py first.")
