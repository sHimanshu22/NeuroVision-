
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
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

classifier = None
class_names = []
classifier_load_error = None


def load_classifier():
    global classifier, class_names, classifier_load_error

    if not CHECKPOINT_PATH.is_file():
        classifier_load_error = (
            f"Classification checkpoint not found: {CHECKPOINT_PATH}"
        )
        print(classifier_load_error)
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

        loaded_model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        loaded_model = loaded_model.to(DEVICE)
        loaded_model.eval()

        classifier = loaded_model
        class_names = checkpoint["class_names"]
        classifier_load_error = None

        print("Classification model loaded.")
        print("Classes:", class_names)
        print("Classification device:", DEVICE)

    except Exception as exc:
        classifier = None
        classifier_load_error = str(exc)
        print(f"Classification model loading failed: {exc}")


def predict_classification(image: Image.Image):
    if classifier is None:
        raise RuntimeError(
            f"Classification model is not ready: "
            f"{classifier_load_error}"
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

    with torch.inference_mode():
        logits = classifier(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(probabilities.argmax().item())

    scores = {
        name: round(float(probabilities[index].item()), 6)
        for index, name in enumerate(class_names)
    }

    return {
        "predicted_class": class_names[predicted_index],
        "confidence": round(
            float(probabilities[predicted_index].item()), 6
        ),
        "class_probabilities": scores,
    }


load_classifier()
