import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unet_model import UNet

ROOT = Path(r"S:\NeuroVision")

image_path = (
    ROOT / "dataset" / "archive (4)" / "brisc2025"
    / "segmentation_task" / "test" / "images"
    / "brisc2025_test_00004_gl_ax_t1.jpg"
)

mask_path = (
    ROOT / "dataset" / "archive (4)" / "brisc2025"
    / "segmentation_task" / "test" / "masks"
    / "brisc2025_test_00004_gl_ax_t1.png"
)

checkpoint_path = (
    ROOT / "ai-service" / "outputs" / "segmentation"
    / "continuation_lr1e4" / "best_continued_unet.pt"
)

device = torch.device("cpu")
torch.set_num_threads(1)
print("Using device:", device)

# Load and preprocess the reference mask.
with Image.open(mask_path) as image:
    reference = image.convert("L").resize(
        (256, 256), Image.Resampling.NEAREST
    )
    ground_truth = np.asarray(reference) >= 128

# Load the MRI using the same preprocessing as the API.
with Image.open(image_path) as image:
    image = image.convert("L").resize(
        (256, 256), Image.Resampling.BILINEAR
    )
    image_array = np.asarray(image, dtype=np.float32).copy() / 255.0

x = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0).to(device)

# Load the existing checkpoint; no training occurs.
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False,
)

model = UNet(in_channels=1, out_channels=1, base_channels=16).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Inference only: do not calculate gradients.
with torch.inference_mode():
    probabilities = torch.sigmoid(model(x))[0, 0]
    prediction = (probabilities >= 0.25).cpu().numpy()

intersection = np.logical_and(prediction, ground_truth).sum()
union = np.logical_or(prediction, ground_truth).sum()
predicted_pixels = int(prediction.sum())
reference_pixels = int(ground_truth.sum())

denominator = predicted_pixels + reference_pixels
dice = 2 * intersection / denominator if denominator else 1.0
iou = intersection / union if union else 1.0

print("\nSingle-image comparison")
print("Predicted pixels:", predicted_pixels)
print("Reference pixels:", reference_pixels)
print("Intersection:", int(intersection))
print("Dice:", round(float(dice), 4))
print("IoU:", round(float(iou), 4))
print("\nDiagnostic only; this does not change the official test evaluation.")