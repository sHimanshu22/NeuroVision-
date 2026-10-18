
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from segmentation_dataset import BrainSegmentationDataset
from unet_model import UNet


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"
METRICS = ROOT / "outputs" / "segmentation" / "validation_case_metrics.csv"
CHECKPOINT = ROOT / "outputs" / "segmentation" / "best_unet.pt"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
THRESHOLDS = [0.10, 0.50]


def dice_score(prediction, target, smooth=1.0):
    intersection = np.logical_and(prediction, target).sum()
    pred_sum = prediction.sum()
    target_sum = target.sum()

    return (
        2 * intersection + smooth
    ) / (
        pred_sum + target_sum + smooth
    )


def main():
    with METRICS.open("r", newline="", encoding="utf-8") as f:
        worst = sorted(
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

    indices = {
        sample["filename"]: i
        for i, sample in enumerate(dataset.samples)
    }

    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=False
    )

    model = UNet(in_channels=1, out_channels=1, base_channels=16)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    print("Threshold comparison on the 10 worst validation cases")

    with torch.inference_mode():
        for record in worst:
            filename = record["filename"]
            sample = dataset[indices[filename]]

            image = sample["image"].unsqueeze(0).to(DEVICE)
            target = sample["mask"][0].numpy().astype(bool)

            probability = torch.sigmoid(model(image))[0, 0].cpu().numpy()

            print(f"\n{filename} | GT pixels={target.sum()}")

            for threshold in THRESHOLDS:
                prediction = probability >= threshold
                dice = dice_score(prediction, target)

                print(
                    f"  threshold={threshold:.2f} | "
                    f"Dice={dice:.4f} | "
                    f"predicted pixels={prediction.sum()}"
                )


if __name__ == "__main__":
    main()
