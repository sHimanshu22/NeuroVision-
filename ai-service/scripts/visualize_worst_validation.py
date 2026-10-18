
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from segmentation_dataset import BrainSegmentationDataset
from unet_model import UNet


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"
CHECKPOINT = ROOT / "outputs" / "segmentation" / "best_unet.pt"
METRICS = ROOT / "outputs" / "segmentation" / "validation_case_metrics.csv"
OUTPUT = ROOT / "outputs" / "segmentation" / "worst_validation_cases.png"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    with METRICS.open("r", newline="", encoding="utf-8") as f:
        records = sorted(
            csv.DictReader(f),
            key=lambda row: float(row["dice"])
        )[:10]

    dataset = BrainSegmentationDataset(
        manifest_path=MANIFEST,
        split="validation",
        image_size=256,
        mask_threshold=128,
        validation_fraction=0.10,
        seed=42,
    )

    index_by_name = {
        sample["filename"]: i
        for i, sample in enumerate(dataset.samples)
    }

    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=False
    )

    model = UNet(in_channels=1, out_channels=1, base_channels=16)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    fig, axes = plt.subplots(10, 4, figsize=(14, 30), squeeze=False)

    with torch.inference_mode():
        for row_index, record in enumerate(records):
            filename = record["filename"]

            if filename not in index_by_name:
                raise RuntimeError(
                    f"Validation sample not found: {filename}"
                )

            sample = dataset[index_by_name[filename]]
            image = sample["image"].unsqueeze(0).to(DEVICE)
            target = sample["mask"][0].numpy().astype(bool)

            probability = torch.sigmoid(model(image))[0, 0].cpu().numpy()
            prediction = probability >= 0.5

            mri = sample["image"][0].numpy()

            # Red = prediction; green = ground truth; yellow = overlap.
            axes[row_index, 0].imshow(mri, cmap="gray")
            axes[row_index, 0].set_title(filename, fontsize=8)

            axes[row_index, 1].imshow(mri, cmap="gray")
            axes[row_index, 1].imshow(
                target, cmap="Greens", alpha=0.65, vmin=0, vmax=1
            )
            axes[row_index, 1].set_title("Ground truth")

            axes[row_index, 2].imshow(mri, cmap="gray")
            axes[row_index, 2].imshow(
                prediction, cmap="Reds", alpha=0.65, vmin=0, vmax=1
            )
            axes[row_index, 2].set_title("Prediction")

            comparison = np.zeros((*target.shape, 3), dtype=np.float32)
            comparison[..., 0] = prediction
            comparison[..., 1] = target

            axes[row_index, 3].imshow(comparison)
            axes[row_index, 3].set_title(
                f"Dice {float(record['dice']):.4f} | "
                f"IoU {float(record['iou']):.4f}"
            )

            for ax in axes[row_index]:
                ax.axis("off")

    fig.suptitle("NeuroVision: Ten Worst Validation Cases", fontsize=15)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
