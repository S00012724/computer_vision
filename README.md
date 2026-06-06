# Waste-bin collection decision system

Collection trucks waste time and fuel when they stop at bins that still have room. This project uses a bin photo to predict one of two labels:

- `has_space`: the bin can wait
- `needs_collection`: the bin should be collected

The API maps `P(needs_collection)` to `urgent`, `low`, or `review` so operators can separate clear cases from images that need a human check. The technical report is available as [report.pdf](report.pdf).

## Project overview

The application wraps the classifier in a FastAPI endpoint and a Gradio upload page. It keeps the model scores visible, then adds a routing policy so operators can review uncertain cases.

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

The response includes the predicted label, class scores, priority, risk level, reason, and processing time.

## Train and evaluate

Train the model:

```bash
python -m src.train
```

Evaluate the saved model:

```bash
python -m src.evaluate
```

Run the comparison baseline:

```bash
python -m src.compare.models
```

The comparison baseline is implemented in [src/compare/data.py](src/compare/data.py) and evaluated from [src/compare/models.py](src/compare/models.py). It uses OpenCV HOG features with a balanced Linear SVM as a reference against the deployed CNN.

## Pipeline and architecture

1. Data loading: [dataset/annotations.json](dataset/annotations.json) maps each image to `is_empty` or `is_full`
2. Label mapping: `is_empty` becomes `has_space`; `is_full` becomes `needs_collection`
3. Preprocessing: OpenCV reads each image, converts BGR to RGB, resizes it to `96 x 96`, and scales pixels from `0` to `1`
4. Feature representation: ResNet50, pretrained on ImageNet, extracts visual features
5. Classification: a custom two-class softmax head predicts `has_space` or `needs_collection`
6. Postprocessing: the `needs_collection` score becomes the collection risk score
7. Routing: risk `>= 0.70` becomes `urgent`, risk `<= 0.15` becomes `low`, and the middle band goes to `review`
8. Deployment: [src/app.py](src/app.py) serves `POST /predict` and a Gradio upload page

Model architecture:

1. Input: `96 x 96 x 3` RGB image
2. ResNet50 preprocessing
3. Frozen ResNet50 backbone
4. Global average pooling
5. Dropout `0.3`
6. Dense softmax output with two classes

## Results summary

The model was evaluated on a stratified 80/20 hold-out split.

| Split | Images | `has_space` | `needs_collection` |
| --- | ---: | ---: | ---: |
| Train | 4,524 | 3,015 | 1,509 |
| Test | 1,130 | 753 | 377 |

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8788 |
| Macro Precision | 0.8725 |
| Macro Recall | 0.8501 |
| Macro F1 | 0.8596 |

Comparison on the same split:

| Model | Features | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| ResNet50 transfer CNN | RGB `96 x 96` pixels | 0.8788 | 0.8725 | 0.8501 | 0.8596 |
| OpenCV HOG 64x64 + Linear SVM | OpenCV HOG `64 x 64` | 0.7062 | 0.6724 | 0.6776 | 0.6746 |

Confusion matrix:

| Ground truth / Prediction | `has_space` | `needs_collection` |
| --- | ---: | ---: |
| `has_space` | 705 | 48 |
| `needs_collection` | 89 | 288 |

Postprocessing uses `P(needs_collection)` as the risk score. It marks 289 test predictions as `urgent`, 623 as `low`, and sends 218 of 1,130 predictions to `review`. Accuracy on the remaining automatic predictions is 0.9298, with 42 full-bin cases still missed automatically.

## Limitations

The model assumes that uploaded images contain waste bins. It does not reject non-bin images, unusual viewpoints, severe blur, or night scenes. A stronger evaluation should split data by bin, location, or image sequence.
