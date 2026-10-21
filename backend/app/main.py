
from pathlib import Path
import io
import sys

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from backend.app.classification_service import (
    predict_classification,
    load_classifier,
    classifier,
    classifier_load_error,
)

from backend.app.gradcam_service import generate_gradcam
from backend.app.gemini_service import (
    GeminiAnalysisRequest,
    GeminiExplanation,
    GeminiInvalidResponseError,
    GeminiNotConfiguredError,
    GeminiProviderError,
    generate_gemini_explanation,
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_SCRIPTS = PROJECT_ROOT / "ai-service" / "scripts"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "ai-service"
    / "outputs"
    / "segmentation"
    / "continuation_lr1e4"
    / "best_continued_unet.pt"
)

# Reuse the existing trained model architecture.
sys.path.insert(0, str(AI_SCRIPTS))
from unet_model import UNet

IMAGE_SIZE = 256
PREDICTION_THRESHOLD = 0.25
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
REFERENCE_FILENAME = "brisc2025_test_00004_gl_ax_t1.jpg"

REFERENCE_MASK_PATH = Path(
    r"S:\NeuroVision\dataset\archive (4)\brisc2025"
    r"\segmentation_task\test\masks"
    r"\brisc2025_test_00004_gl_ax_t1.png"
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="NeuroVision API",
    description="Research prototype for brain MRI segmentation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

model = None
model_load_error = None


def load_model():
    global model, model_load_error

    if not CHECKPOINT_PATH.is_file():
        model_load_error = f"Checkpoint not found: {CHECKPOINT_PATH}"
        return

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False,
        )

        loaded_model = UNet(
            in_channels=1,
            out_channels=1,
            base_channels=16,
        ).to(DEVICE)

        loaded_model.load_state_dict(checkpoint["model_state_dict"])
        loaded_model.eval()

        model = loaded_model
        model_load_error = None
        print(f"Loaded checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Using device: {DEVICE}")

    except Exception as exc:
        model = None
        model_load_error = str(exc)
        print(f"Model loading failed: {exc}")


load_model()


@app.get("/")
def root():
    return {
        "name": "NeuroVision API",
        "status": "running",
        "model_loaded": model is not None,
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "image_size": IMAGE_SIZE,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "model_error": model_load_error,
    }

@app.get("/api/reference-mask")
def get_reference_mask(filename: str):
    if filename != REFERENCE_FILENAME:
        raise HTTPException(
            status_code=404,
            detail="No reference mask is configured for this example.",
        )

    if not REFERENCE_MASK_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail="Reference mask file was not found.",
        )

    with Image.open(REFERENCE_MASK_PATH) as reference_image:
        reference_image = reference_image.convert("L")
        reference_image = TF.resize(
            reference_image,
            [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.NEAREST,
        )

        import numpy as np
        import base64

        reference_array = np.array(reference_image)
        binary_mask = (reference_array >= 128).astype("uint8") * 255

        output = io.BytesIO()
        Image.fromarray(binary_mask, mode="L").save(output, format="PNG")

    return {
        "filename": filename,
        "mask_png_base64": base64.b64encode(
            output.getvalue()
        ).decode("ascii"),
        "note": "Reference mask for local qualitative inspection only.",
    }

@app.post("/api/segment")
async def segment_mri(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not ready: {model_load_error}",
        )

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Upload an image file.",
        )

    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)

        if not contents:
            raise HTTPException(status_code=400, detail="The file is empty.")

        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image must be 10 MB or smaller.",
            )

        with Image.open(io.BytesIO(contents)) as uploaded_image:
            image = uploaded_image.convert("L")

        original_width, original_height = image.size

        image = TF.resize(
            image,
            [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.BILINEAR,
        )

        image_tensor = TF.to_tensor(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            logits = model(image_tensor)
            probabilities = torch.sigmoid(logits)
            binary_mask = (
                probabilities >= PREDICTION_THRESHOLD
            ).squeeze().to(torch.uint8).cpu().numpy()

        # Encode the predicted binary mask as a PNG image.
        mask_image = Image.fromarray(binary_mask * 255, mode="L")

        # Return the mask and a simple foreground-pixel summary.
        output_buffer = io.BytesIO()
        mask_image.save(output_buffer, format="PNG")

        import base64

        return {
            "success": True,
            "filename": file.filename,
            "original_width": original_width,
            "original_height": original_height,
            "processed_size": IMAGE_SIZE,
            "threshold": PREDICTION_THRESHOLD,
            "foreground_pixels": int(binary_mask.sum()),
            "mask_png_base64": base64.b64encode(
                output_buffer.getvalue()
            ).decode("ascii"),
            "notice": (
                "Research prototype only. This segmentation output "
                "is not a medical diagnosis."
            ),
        }

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Could not read this image. Please upload a valid image.",
        )
    except Exception as exc:
        print(f"Inference error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Segmentation failed. Check the backend terminal.",
        )
    finally:
        await file.close()

@app.post("/api/classify")
async def classify_mri(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Upload an image file.",
        )

    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The file is empty.",
            )

        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image must be 10 MB or smaller.",
            )

        with Image.open(io.BytesIO(contents)) as uploaded_image:
            image = uploaded_image.copy()

        result = predict_classification(image)

        return {
            "success": True,
            "filename": file.filename,
            **result,
            "notice": (
                "Research prototype only. This output is not "
                "a medical diagnosis."
            ),
        }

    except RuntimeError as exc:
        print(f"Classification error: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Classification model is not ready.",
        )

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Could not read this image. Please upload a valid image.",
        )

    except Exception as exc:
        print(f"Classification inference error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Classification failed. Check the backend terminal.",
        )

    finally:
        await file.close()


@app.post("/api/gradcam")
async def gradcam_mri(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Upload an image file.",
        )

    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The file is empty.",
            )

        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image must be 10 MB or smaller.",
            )

        with Image.open(io.BytesIO(contents)) as uploaded_image:
            image = uploaded_image.copy()

        result = generate_gradcam(image)

        return {
            "success": True,
            "filename": file.filename,
            **result,
        }

    except RuntimeError as exc:
        print(f"Grad-CAM error: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Grad-CAM model is not ready.",
        )

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Could not read this image. Please upload a valid image.",
        )

    except Exception as exc:
        print(f"Grad-CAM inference error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Grad-CAM generation failed. Check the backend terminal.",
        )

    finally:
        await file.close()


@app.post("/api/gemini-explanation", response_model=GeminiExplanation)
async def gemini_explanation(request: GeminiAnalysisRequest):
    try:
        return generate_gemini_explanation(request)
    except GeminiNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except GeminiInvalidResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None
    except GeminiProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
