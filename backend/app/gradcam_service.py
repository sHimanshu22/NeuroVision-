
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "ai-service"
    / "outputs"
    / "classification"
    / "best_resnet18.pt"
)

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

class_names = []
model = None
model_load_error = None


def load_gradcam_model():
    global model, class_names, model_load_error

    if not CHECKPOINT_PATH.is_file():
        model_load_error = f"Checkpoint not found: {CHECKPOINT_PATH}"
        print(model_load_error)
        return

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=True,
        )

        loaded_model = models.resnet18(weights=None)
        loaded_model.fc = nn.Linear(
            loaded_model.fc.in_features,
            len(checkpoint["class_names"]),
        )

        loaded_model.load_state_dict(checkpoint["model_state_dict"])
        loaded_model = loaded_model.to(DEVICE)
        loaded_model.eval()

        model = loaded_model
        class_names = checkpoint["class_names"]
        model_load_error = None

        print("Grad-CAM model loaded.")
        print("Grad-CAM device:", DEVICE)

    except Exception as exc:
        model = None
        model_load_error = str(exc)
        print("Grad-CAM model loading failed:", exc)


def generate_gradcam(image: Image.Image):
    if model is None:
        raise RuntimeError(
            f"Grad-CAM model is not ready: {model_load_error}"
        )

    image = image.convert("L")
    image = TF.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        interpolation=InterpolationMode.BILINEAR,
    )
    image = image.convert("RGB")

    tensor = TF.to_tensor(image).unsqueeze(0)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = ((tensor - mean) / std).to(DEVICE)

    activations = []
    gradients = []

    def save_activation(module, inputs, output):
        activations.append(output)

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.layer4[-1]

    activation_handle = target_layer.register_forward_hook(
        save_activation
    )
    gradient_handle = target_layer.register_full_backward_hook(
        save_gradient
    )

    try:
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = model(tensor)
            predicted_index = int(logits.argmax(dim=1).item())
            score = logits[0, predicted_index]
            score.backward()

        feature_maps = activations[0]
        feature_gradients = gradients[0]

        weights = feature_gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (weights * feature_maps).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].detach().cpu().numpy()

        minimum = float(cam.min())
        maximum = float(cam.max())

        if maximum > minimum:
            cam = (cam - minimum) / (maximum - minimum)
        else:
            cam = np.zeros_like(cam)

        heatmap = np.uint8(cam * 255)

        heatmap_image = Image.fromarray(heatmap, mode="L")
        output = io.BytesIO()
        heatmap_image.save(output, format="PNG")

        probabilities = torch.softmax(logits.detach(), dim=1)[0]

        return {
            "predicted_class": class_names[predicted_index],
            "confidence": round(
                float(probabilities[predicted_index].item()), 6
            ),
            "heatmap_png_base64": __import__("base64").b64encode(
                output.getvalue()
            ).decode("ascii"),
            "notice": (
                "Grad-CAM is a model-attention visualization, "
                "not a verified tumor boundary or medical diagnosis."
            ),
        }

    finally:
        activation_handle.remove()
        gradient_handle.remove()


load_gradcam_model()
