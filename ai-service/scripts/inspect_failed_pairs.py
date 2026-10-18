
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "segmentation" / "validation_case_metrics.csv"
OUTPUT = ROOT / "outputs" / "segmentation" / "failed_pair_inspection.png"

with METRICS.open("r", newline="", encoding="utf-8") as f:
    records = sorted(
        csv.DictReader(f),
        key=lambda row: float(row["dice"])
    )[:3]

fig, axes = plt.subplots(len(records), 3, figsize=(12, 4 * len(records)))

if len(records) == 1:
    axes = np.array([axes])

for i, record in enumerate(records):
    filename = record["filename"]

    # Reconstruct the image and mask paths using the dataset structure.
    split = "train"
    dataset_root = Path(
        r"S:\NeuroVision\dataset\archive (4)\brisc2025"
    )

    image_path = (
        dataset_root / "segmentation_task" / split / "images" / filename
    )
    mask_path = (
        dataset_root / "segmentation_task" / split / "masks"
        / f"{Path(filename).stem}.png"
    )

    if not image_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(
            f"Missing pair for {filename}\n"
            f"Image: {image_path}\nMask: {mask_path}"
        )

    image = np.array(Image.open(image_path).convert("L"))
    mask = np.array(Image.open(mask_path).convert("L"))

    if image.shape != mask.shape:
        raise ValueError(
            f"Dimension mismatch for {filename}: "
            f"image={image.shape}, mask={mask.shape}"
        )

    binary_mask = mask >= 128

    axes[i, 0].imshow(image, cmap="gray")
    axes[i, 0].set_title(f"Original MRI\n{filename}", fontsize=8)

    axes[i, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
    axes[i, 1].set_title("Original mask")

    axes[i, 2].imshow(image, cmap="gray")
    axes[i, 2].imshow(binary_mask, cmap="Greens", alpha=0.5)
    axes[i, 2].set_title(
        f"Original-resolution overlay\n"
        f"Dice={float(record['dice']):.4f}"
    )

    for ax in axes[i]:
        ax.axis("off")

    print(
        f"{filename}: image={image.shape}, mask={mask.shape}, "
        f"foreground pixels={int(binary_mask.sum())}"
    )

fig.tight_layout()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nSaved:", OUTPUT)
