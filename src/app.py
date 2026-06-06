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


# Loaded once when FastAPI starts. Keeping it global avoids loading the Keras
# model again for every request, which would make predictions very slow
MODEL = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the trained model into memory on startup

    Args:
        _: FastAPI app instance supplied by the lifespan hook

    Yields:
        Control to FastAPI while the app is running
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
    """Round one API score while preserving missing values

    Args:
        value: Score returned by the model or None when unavailable

    Returns:
        Rounded score or None
    """

    if value is None:
        return None
    return round(float(value), 4)


def _round_scores(class_scores: dict[str, float]) -> dict[str, float]:
    """Round every class score for a stable JSON response

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
    """Group model output, routing decision, and request metadata

    Args:
        class_name: Predicted class label
        confidence: Probability assigned to the predicted class
        class_scores: Mapping from class name to model probability
        postprocessed: Routing decision returned by postprocess_prediction
        processing_time_seconds: Time spent on prediction and post-processing

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
    """Map a model prediction index to its output label

    Args:
        prediction_index: Integer class index returned by the model
        class_names: Ordered class labels used during training

    Returns:
        Class label for the prediction index
    """
    return class_names[int(prediction_index)]


def _cpu_bound_prediction(image_bgr: np.ndarray):
    """Run image preprocessing and CNN prediction

    FastAPI calls this in a thread pool because TensorFlow prediction is CPU/GPU
    work, so the async endpoint can keep handling request I/O

    Args:
        image_bgr: OpenCV image in BGR channel order

    Returns:
        Prediction index, predicted-class confidence, and per-class scores
    """

    tensor = image_to_cnn_tensor(
        image_bgr,
        image_size=CNN_IMAGE_SIZE,
    ).reshape(1, *CNN_IMAGE_SIZE, 3)
    probabilities = MODEL.predict(tensor, verbose=0)[0]
    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction])
    scores = {
        class_name: float(probability)
        for class_name, probability in zip(CLASS_NAMES, probabilities)
    }
    return prediction, confidence, scores


def predict_bgr_image(image_bgr: np.ndarray):
    """Run prediction plus post-processing for one OpenCV BGR image

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
    """Predict the bin fill level from a Gradio image input

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
    """Predict bin fill level from one uploaded image

    Args:
        file: Uploaded image file received by FastAPI

    Returns:
        JSON response with prediction, decision, and metadata
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        contents = await file.read()
        image_array = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        result = await run_in_threadpool(predict_bgr_image, image_bgr)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


with gr.Blocks(title="Waste-bin Fill-Level Detection") as gradio_app:
    gr.Markdown("# Waste-bin Fill-Level Detection")
    gr.Markdown("Upload a waste-bin image to check whether it has space or needs collection.")
    gr.Markdown("**Note:** This model only analyzes waste bins. It cannot process arbitrary images.")
    image_input = gr.Image(type="numpy", label="Waste-bin image")
    analyze_button = gr.Button("Analyze Image")
    summary_output = gr.Textbox(label="Result", lines=5)
    details_output = gr.JSON(label="Raw API-style response")
    analyze_button.click(
        fn=predict_image_array,
        inputs=image_input,
        outputs=[summary_output, details_output],
    )


app = gr.mount_gradio_app(app, gradio_app, path="/")
