import logging
import time
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from src.config import CNN_IMAGE_SIZE, CNN_MODEL_PATH
from src.data import CLASS_NAMES
from src.postprocessing import postprocess_prediction
from src.preprocessing import image_to_cnn_tensor


# FastAPI loads the model at startup and reuses it for every request. Loading
# the Keras file per request would make prediction much slower
MODEL = None
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_CONTENT_TYPES = frozenset({
    "image/bmp",
    "image/jpeg",
    "image/png",
    "image/webp",
})
LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the trained model when FastAPI starts

    Args:
        _: FastAPI app supplied by the lifespan hook

    Yields:
        Control back to FastAPI while the app runs
    """
    global MODEL
    model_path = Path(CNN_MODEL_PATH)
    try:
        if not model_path.is_file():
            raise FileNotFoundError(f"CNN model not found: {model_path}")

        MODEL = tf.keras.models.load_model(model_path)
        print(f"CNN model loaded successfully from {model_path}")
    except Exception as error:
        print(f"Warning: Could not load model. Error: {error}")
    yield
    MODEL = None


app = FastAPI(
    title="Waste Management API",
    description="API for waste-bin fill-level prediction.",
    version="1.0.0",
    lifespan=lifespan,
)


def _round_score(value: float | None) -> float | None:
    """Round one API score without changing missing values

    Args:
        value: Score returned by the model or None when unavailable

    Returns:
        Rounded score or None
    """

    if value is None:
        return None
    return round(float(value), 4)


def _round_scores(class_scores: dict[str, float]) -> dict[str, float]:
    """Round class scores for stable JSON output

    Args:
        class_scores: Mapping from class name to model probability

    Returns:
        New mapping with rounded score values
    """

    return {
        class_name: round(float(score), 4)
        for class_name, score in class_scores.items()
    }


def build_prediction_response(
    class_name: str,
    confidence: float | None,
    class_scores: dict[str, float],
    postprocessed: dict[str, object],
    processing_time_seconds: float,
) -> dict[str, object]:
    """Build the API response from prediction and routing data

    Args:
        class_name: Predicted class label
        confidence: Probability assigned to the predicted class
        class_scores: Mapping from class name to model probability
        postprocessed: Routing decision from postprocess_prediction
        processing_time_seconds: Time spent on prediction and postprocessing

    Returns:
        API response payload with prediction, decision, and metadata sections
    """
    return {
        "prediction": {
            "label": class_name,
            "confidence": _round_score(confidence),
            "scores": _round_scores(class_scores),
        },
        "decision": {
            "priority": postprocessed["collection_priority"],
            "risk_level": postprocessed["risk_level"],
            "reason": postprocessed["priority_reason"],
        },
        "metadata": {
            "processing_time_seconds": round(processing_time_seconds, 4),
        },
    }


def label_prediction(prediction_index: int, class_names: Sequence[str]) -> str:
    """Map a model prediction index to an output label

    Args:
        prediction_index: Integer class index returned by the model
        class_names: Ordered class labels used during training

    Returns:
        Class label for the prediction index
    """
    return class_names[int(prediction_index)]


def _cpu_bound_prediction(image_bgr: np.ndarray):
    """Preprocess one image and run the CNN

    FastAPI runs this in a thread pool so TensorFlow work does not block
    request I/O

    Args:
        image_bgr: OpenCV image in BGR channel order

    Returns:
        Prediction index, predicted-class confidence, and per-class scores
    """

    tensor = image_to_cnn_tensor(
        image_bgr,
        image_size=CNN_IMAGE_SIZE,
    ).reshape(1, *CNN_IMAGE_SIZE, 3)
    model_output = MODEL(tensor, training=False)
    probabilities = np.asarray(model_output.numpy())[0]
    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction])
    scores = {
        class_name: float(probability)
        for class_name, probability in zip(CLASS_NAMES, probabilities)
    }
    return prediction, confidence, scores


def predict_bgr_image(image_bgr: np.ndarray):
    """Run prediction and postprocessing for one OpenCV BGR image

    Args:
        image_bgr: OpenCV image in BGR channel order

    Returns:
        API response payload for the image
    """

    start_time = time.perf_counter()
    prediction, confidence, class_scores = _cpu_bound_prediction(image_bgr)
    class_name = label_prediction(prediction, CLASS_NAMES)
    postprocessed = postprocess_prediction(
        class_name,
        confidence=confidence,
        class_scores=class_scores,
    )

    return build_prediction_response(
        class_name,
        confidence=confidence,
        class_scores=class_scores,
        postprocessed=postprocessed,
        processing_time_seconds=time.perf_counter() - start_time,
    )


def predict_image_array(image):
    """Predict bin fill level from a Gradio image input

    Args:
        image: RGB NumPy array provided by the Gradio image component

    Returns:
        Human-readable summary and raw API-style response data
    """
    if MODEL is None:
        return "Model is not loaded.", {
            "error": "Model is not loaded.",
        }

    if image is None:
        return "Upload an image first.", {
            "error": "No image uploaded.",
        }

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = predict_bgr_image(image_bgr)

    prediction = result["prediction"]
    decision = result["decision"]
    metadata = result["metadata"]
    class_name = prediction["label"]
    label = "Needs collection" if class_name == "needs_collection" else "Has space"
    confidence = prediction["confidence"]
    confidence_text = "N/A" if confidence is None else f"{confidence * 100:.2f}%"
    summary = (
        f"Prediction: {label}\n"
        f"Confidence: {confidence_text}\n"
        f"Priority: {decision['priority']}\n"
        f"Risk level: {decision['risk_level']}\n"
        f"Reason: {decision['reason']}\n"
        f"Processing time: {metadata['processing_time_seconds']} seconds"
    )
    return summary, result


@app.post("/predict")
async def predict_fill_level(file: UploadFile = File(...)):
    """Predict bin fill level from one uploaded image file

    Args:
        file: Uploaded image file received by FastAPI

    Returns:
        JSON response with prediction, decision, and metadata
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        if file.content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail="Unsupported image type.",
            )

        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the 10 MB upload limit.",
            )

        image_array = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        result = await run_in_threadpool(predict_bgr_image, image_bgr)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as error:
        LOGGER.exception("Prediction failed for upload %r.", file.filename)
        raise HTTPException(status_code=500, detail="Prediction failed.") from error


with gr.Blocks(title="Waste-bin Fill-Level Detection") as gradio_app:
    gr.Markdown("# Waste-bin fill-level detection")
    gr.Markdown("Upload a waste-bin image to check whether it has space or needs collection.")
    gr.Markdown("This model only analyzes waste-bin images. It cannot process arbitrary images.")
    image_input = gr.Image(type="numpy", label="Waste-bin image")
    analyze_button = gr.Button("Analyze image")
    summary_output = gr.Textbox(label="Result", lines=5)
    details_output = gr.JSON(label="Raw API-style response")
    analyze_button.click(
        fn=predict_image_array,
        inputs=image_input,
        outputs=[summary_output, details_output],
    )


app = gr.mount_gradio_app(app, gradio_app, path="/")
