# Image-based waste-bin collection decisions

Collection trucks waste time and fuel when they stop at bins that still have room. This project uses a bin photo to predict one of two labels:

This prototype classifies one waste-bin image as `has_space` or `needs_collection`. A fixed threshold policy then labels the model output as `urgent`, `low`, or `review`. The `review` value is an API label; the application does not implement a review queue or assign cases to an operator.


## Dataset

The images are mix of multiple public datasets. Every image has one fill-level tag:

| Source tag | Project class | Images |
| --- | --- | ---: |
| `is_empty` | `has_space` | 3,768 |
| `is_full` | `needs_collection` | 1,886 |

The repository contains the dataset, source code, saved Keras model, FastAPI and Gradio app, evaluation scripts, and final PDF report.

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the application

```bash
python -m uvicorn src.app:app --reload
```

Open:

- Web app: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

The app loads [models/model.keras](models/model.keras). If the file is missing, run training first.

## API usage

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@dataset/img/134.jpg"
```

The endpoint accepts JPEG, PNG, WebP, and BMP uploads up to 10 MB.

## Train and evaluate

Training writes a new model to [models/model.keras](models/model.keras):

```bash
python -m src.train
```

Evaluate the saved CNN:

```bash
python -m src.evaluate
```

This command prints accuracy, macro precision, macro recall, macro F1, the confusion matrix, and routing-policy results. The CNN metric implementation is in [src/models.py](src/models.py); the HOG comparison has its metric implementation in [src/compare/data.py](src/compare/data.py). Both use scikit-learn.

Run the handcrafted-feature comparison:

```bash
python -m src.compare.models
```

## Split and evaluation scope

The code makes one deterministic 80/20 split within each class. It sorts records by numeric image ID and places the highest-numbered 20% in the evaluation split. This is stratified, but it is not random or group-aware.

The 1,130 evaluation images do not contribute gradient updates, but `model.fit` uses them as validation data. The CNN comparison and routing-threshold analysis also use this split. The numbers below are therefore exploratory evaluation results, not measurements from an untouched final test set. A stronger experiment needs separate train, validation, and group-based test splits.

| Split | Images | `has_space` | `needs_collection` |
| --- | ---: | ---: | ---: |
| Train | 4,524 | 3,015 | 1,509 |
| Evaluation | 1,130 | 753 | 377 |

## Pipeline and architecture

1. [src/data.py](src/data.py) reads the JSON annotations and maps `is_empty` to `has_space` and `is_full` to `needs_collection`.
2. [src/preprocessing.py](src/preprocessing.py) converts OpenCV BGR images to RGB, resizes them to `96 x 96`, and scales pixels to the range 0 to 1.
3. A frozen ResNet50 backbone initialized with ImageNet weights extracts image features.
4. Global average pooling, dropout at 0.3, and a two-unit softmax layer produce the class scores.
5. [src/postprocessing.py](src/postprocessing.py) applies fixed thresholds to the `needs_collection` softmax score.
6. [src/app.py](src/app.py) serves `POST /predict` and mounts the Gradio page at the application root.

The softmax scores have not been calibrated. They should not be read as measured real-world probabilities.

## Results

The saved CNN reproduces the following scikit-learn metrics on the evaluation split:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8788 |
| Macro Precision | 0.8725 |
| Macro Recall | 0.8501 |
| Macro F1 | 0.8596 |

The HOG values below were reproduced with the pinned environment in [requirements.txt](requirements.txt).

| Model | Input or features | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| ResNet50 transfer CNN | RGB `96 x 96` pixels | 0.8788 | 0.8725 | 0.8501 | 0.8596 |
| OpenCV HOG + Linear SVM | HOG from `64 x 64` grayscale images | 0.7053 | 0.6713 | 0.6762 | 0.6734 |

![Scikit-learn metric summary](report/assets/metric_summary.png)

![Confusion matrix](report/assets/confusion_matrix.png)

The CNN correctly classifies 705 of 753 `has_space` images and 288 of 377 `needs_collection` images. It produces 48 false collection alarms and misses 89 full-bin cases.

## Decision routing

The routing thresholds are heuristic. A `needs_collection` softmax score at or above 0.70 produces `urgent` priority and `high` risk. A score at or below 0.15 produces `low` priority and `low` risk. Values between the thresholds produce `review` priority and `medium` risk.

| Route | Evaluation images |
| --- | ---: |
| `urgent` | 289 |
| `review` | 218 |
| `low` | 623 |

The non-review subset contains 912 images and has 0.9298 accuracy. This is selective accuracy at 80.7% coverage, not an improvement to the classifier itself. Of the 377 full-bin images, 267 are `urgent`, 68 are `review`, and 42 are incorrectly routed to `low`. The urgent route therefore captures 70.8% of all full-bin images.

The API reports these paired values as `collection_priority` and `risk_level`. If no `needs_collection` score is available, it returns `review` priority and `unknown` risk.

## Limits and deployment warnings

- The model assumes that every input contains a waste bin. It has no rejection class for unrelated images, severe blur, blocked views, or night scenes.
- Similar images may occur across the current train and evaluation partitions. A split by bin, location, or photo session is needed before making generalization claims.
- The API checks the declared image type, caps uploads at 10 MB, and hides internal exception details. It still has no authentication or rate limit, so it should not be exposed directly to the internet.
- `processing_time_seconds` measures preprocessing, model inference, and routing after decoding. It excludes upload transfer, file reading, image decoding, thread scheduling, and response serialization.
