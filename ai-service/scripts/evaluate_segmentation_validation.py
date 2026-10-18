
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from segmentation_dataset import BrainSegmentationDataset
from unet_model import UNet


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"
CHECKPOINT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "continuation_lr1e4"
    / "best_continued_unet.pt"
)
OUTPUT_CSV = ROOT / "outputs" / "segmentation" / "validation_case_metrics.csv"

IMAGE_SIZE = 256
THRESHOLD = 0.5
BATCH_SIZE = 2


def calculate_metrics(prediction, target, smooth=1.0):
    prediction = prediction.bool()
    target = target.bool()

    intersection = (prediction & target).sum().item()
    predicted_pixels = prediction.sum().item()
    target_pixels = target.sum().item()

    union = predicted_pixels + target_pixels - intersection

    dice = (
        2.0 * intersection + smooth
    ) / (
        predicted_pixels + target_pixels + smooth
    )

    iou = (intersection + smooth) / (union + smooth)

    return dice, iou, predicted_pixels, target_pixels


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    dataset = BrainSegmentationDataset(
        manifest_path=MANIFEST,
        split="validation",
        image_size=IMAGE_SIZE,
        mask_threshold=128,
        validation_fraction=0.10,
        seed=42,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=16,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    records = []

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            probabilities = torch.sigmoid(logits)
            predictions = probabilities >= THRESHOLD

            for i, filename in enumerate(batch["filename"]):
                dice, iou, pred_pixels, target_pixels = calculate_metrics(
                    predictions[i, 0],
                    masks[i, 0] >= 0.5,
                )

                records.append({
                    "filename": filename,
                    "dice": dice,
                    "iou": iou,
                    "predicted_foreground_pixels": pred_pixels,
                    "ground_truth_foreground_pixels": target_pixels,
                })

    if len(records) != len(dataset):
        raise RuntimeError(
            f"Evaluated {len(records)} samples, expected {len(dataset)}."
        )

    dice_values = np.array([r["dice"] for r in records])
    iou_values = np.array([r["iou"] for r in records])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print("\n--- FULL VALIDATION RESULTS ---")
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Checkpoint recorded best Dice:", checkpoint["best_val_dice"])
    print("Validation images evaluated:", len(records))
    print("Mean Dice:", round(float(dice_values.mean()), 4))
    print("Median Dice:", round(float(np.median(dice_values)), 4))
    print("Mean IoU:", round(float(iou_values.mean()), 4))
    print("Median IoU:", round(float(np.median(iou_values)), 4))
    print("Cases with Dice < 0.50:", int((dice_values < 0.50).sum()))
    print("Cases with Dice < 0.70:", int((dice_values < 0.70).sum()))

    print("\n--- 10 WORST VALIDATION CASES ---")
    for record in sorted(records, key=lambda r: r["dice"])[:10]:
        print(
            f"{record['filename']} | "
            f"Dice={record['dice']:.4f} | "
            f"IoU={record['iou']:.4f} | "
            f"Predicted pixels={record['predicted_foreground_pixels']} | "
            f"Ground-truth pixels={record['ground_truth_foreground_pixels']}"
        )

    print("\nPer-case metrics saved to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
