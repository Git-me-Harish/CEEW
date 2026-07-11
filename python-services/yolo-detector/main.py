"""
PashuMitra YOLO Detector — FastAPI micro-service
Loads version4.pt (trained YOLO model for Indian bovine breed classification)
and exposes a /detect endpoint that returns detected breeds with confidence.

Expected model file path: ./models/version4.pt
If model file is missing, the service starts in 'unavailable' mode and /detect
returns a 503 with a clear message so the Next.js side can fall back to VLM-only.
"""

import io
import os
import base64
import logging
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yolo-detector")

MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", os.path.join(os.path.dirname(__file__), "models", "version4.pt"))
PORT = int(os.environ.get("PORT", "8501"))

app = FastAPI(title="PashuMitra YOLO Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model loading ----------------------------------------------------------
yolo_model = None
model_names: dict = {}
model_load_error: Optional[str] = None


def load_model() -> None:
    """Load the YOLO model if available. Sets module-level yolo_model and model_names."""
    global yolo_model, model_names, model_load_error
    if not os.path.exists(MODEL_PATH):
        model_load_error = f"Model file not found at {MODEL_PATH}. Please place version4.pt there."
        logger.warning(model_load_error)
        return
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(MODEL_PATH)
        # Force CPU (sandbox has no GPU)
        try:
            yolo_model.to("cpu")
        except Exception:
            pass
        model_names = dict(yolo_model.names) if hasattr(yolo_model, "names") else {}
        logger.info(f"YOLO model loaded. {len(model_names)} classes: {list(model_names.values())[:8]}...")
    except Exception as e:
        model_load_error = f"Failed to load YOLO model: {e}"
        logger.error(model_load_error, exc_info=True)


# Load on startup
load_model()


# --- Routes -----------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list
    error: Optional[str] = None


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — also reports whether the YOLO model is loaded."""
    return HealthResponse(
        status="ok" if yolo_model is not None else "degraded",
        model_loaded=yolo_model is not None,
        classes=list(model_names.values()),
        error=model_load_error,
    )


@app.post("/detect")
async def detect(image: UploadFile = File(...), conf: float = 0.25, iou: float = 0.45):
    """
    Run YOLO detection on an uploaded image.
    Returns detected objects with class names and confidence scores.
    Also returns a base64-encoded annotated image (with bounding boxes drawn).
    """
    if yolo_model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "YOLO model not loaded",
                "message": model_load_error or "Model unavailable. Please place version4.pt in the models/ directory.",
            },
        )

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run inference
        results = yolo_model.predict(
            source=img,
            conf=conf,
            iou=iou,
            show_labels=True,
            show_conf=True,
            imgsz=640,
            device="cpu",
            verbose=False,
        )

        detected_objects = []
        annotated_image_b64 = None

        for r in results:
            # Generate annotated image with bboxes drawn
            try:
                im_array = r.plot()
                annotated = Image.fromarray(im_array[..., ::-1])
                buf = io.BytesIO()
                annotated.save(buf, format="JPEG", quality=85)
                annotated_image_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            except Exception as e:
                logger.warning(f"Could not generate annotated image: {e}")

            for box in r.boxes:
                cls_id = int(box.cls)
                detected_objects.append({
                    "class": model_names.get(cls_id, f"class_{cls_id}"),
                    "classId": cls_id,
                    "confidence": round(float(box.conf), 4),
                    "bbox": [round(float(x), 2) for x in box.xyxy[0].tolist()],
                })

        # Sort by confidence descending
        detected_objects.sort(key=lambda x: x["confidence"], reverse=True)

        # Primary breed = highest confidence detection
        primary = detected_objects[0] if detected_objects else None

        return JSONResponse({
            "success": True,
            "primary": primary,
            "allDetections": detected_objects,
            "annotatedImage": annotated_image_b64,
            "modelLoaded": True,
            "classesAvailable": list(model_names.values()),
            "thresholds": {"conf": conf, "iou": iou},
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.get("/")
async def root():
    return {
        "service": "PashuMitra YOLO Detector",
        "version": "1.0.0",
        "modelLoaded": yolo_model is not None,
        "endpoints": ["/health", "/detect"],
        "modelPath": MODEL_PATH,
        "modelExists": os.path.exists(MODEL_PATH),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
