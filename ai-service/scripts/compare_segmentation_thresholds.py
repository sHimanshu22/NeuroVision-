
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from segmentation_dataset import BrainSegmentationDataset
from unet_model import UNet


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"
CHECKPOINT = ROOT / "outputs" / "segmentation" / "best_unet.pt"

THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    dataset = BrainSegmentationDataset(
        manifest_path=MANIFEST,
        split="validation",
        image_size=256,
        mask_threshold=128,
        validation_fraction=0.10,
        seed=42,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    model = UNet(in_channels=1, out_channels=1, base_channels=16)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    dice_sums = {t: 0.0 for t in THRESHOLDS}
    iou_sums = {t: 0.0 for t in THRESHOLDS}
    total = 0

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(DEVICE)
            masks = batch["mask"][:, 0].to(DEVICE).bool()

            probabilities = torch.sigmoid(model(images))[:, 0]

            for threshold in THRESHOLDS:
                predictions = probabilities >= threshold

                intersection = (predictions & masks).sum(dim=(1, 2)).float()
                pred_sum = predictions.sum(dim=(1, 2)).float()
                target_sum = masks.sum(dim=(1, 2)).float()

                dice = (
                    2 * intersection + 1.0
                ) / (pred_sum + target_sum + 1.0)

                union = pred_sum + target_sum - intersection
                iou = (intersection + 1.0) / (union + 1.0)

                dice_sums[threshold] += dice.sum().item()
                iou_sums[threshold] += iou.sum().item()

            total += images.size(0)

    print("\nValidation threshold comparison")
    print("Samples:", total)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("\nThreshold | Mean Dice | Mean IoU")

    for threshold in THRESHOLDS:
        print(
            f"{threshold:9.2f} | "
            f"{dice_sums[threshold] / total:.4f}    | "
            f"{iou_sums[threshold] / total:.4f}"
        )


if __name__ == "__main__":
    main()
