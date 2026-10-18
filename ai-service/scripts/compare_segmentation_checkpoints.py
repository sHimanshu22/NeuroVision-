
import csv
from pathlib import Path

import numpy as np
import torch

from segmentation_dataset import BrainSegmentationDataset
from unet_model import UNet


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "data" / "segmentation_manifest.csv"
BASELINE = ROOT / "outputs" / "segmentation" / "best_unet.pt"
CONTINUED = (
    ROOT / "outputs" / "segmentation" / "continuation_lr1e4"
    / "best_continued_unet.pt"
)

OUTPUT = (
    ROOT / "outputs" / "segmentation"
    / "checkpoint_comparison.csv"
)

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


def load_model(path):
    checkpoint = torch.load(
        path, map_location=DEVICE, weights_only=False
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=16,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def main():
    dataset = BrainSegmentationDataset(
        manifest_path=MANIFEST,
        split="validation",
        image_size=256,
        mask_threshold=128,
        validation_fraction=0.10,
        seed=42,
    )

    baseline, base_ckpt = load_model(BASELINE)
    continued, cont_ckpt = load_model(CONTINUED)

    print("Baseline checkpoint epoch:", base_ckpt["epoch"])
    print("Continuation checkpoint epoch:", cont_ckpt["epoch"])
    print("Validation samples:", len(dataset))

    rows = []

    with torch.inference_mode():
        for sample in dataset:
            image = sample["image"].unsqueeze(0).to(DEVICE)
            target = sample["mask"][0].cpu().numpy() >= 0.5

            record = {"filename": sample["filename"]}

            for name, model in [
                ("baseline", baseline),
                ("continued", continued),
            ]:
                probability = torch.sigmoid(model(image))[0, 0]
                prediction = probability.cpu().numpy() >= 0.5

                intersection = np.logical_and(
                    prediction, target
                ).sum()

                pred_sum = prediction.sum()
                target_sum = target.sum()
                union = pred_sum + target_sum - intersection

                dice = (
                    2.0 * intersection + 1.0
                ) / (pred_sum + target_sum + 1.0)

                iou = (intersection + 1.0) / (union + 1.0)

                record[f"{name}_dice"] = float(dice)
                record[f"{name}_iou"] = float(iou)
                record[f"{name}_pred_pixels"] = int(pred_sum)

            rows.append(record)

    fields = [
        "filename",
        "baseline_dice",
        "continued_dice",
        "baseline_iou",
        "continued_iou",
        "baseline_pred_pixels",
        "continued_pred_pixels",
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    for name in ["baseline", "continued"]:
        dice = np.array([r[f"{name}_dice"] for r in rows])
        iou = np.array([r[f"{name}_iou"] for r in rows])
        pred = np.array([r[f"{name}_pred_pixels"] for r in rows])

        print(f"\n{name.upper()}")
        print(f"Mean Dice: {dice.mean():.4f}")
        print(f"Median Dice: {np.median(dice):.4f}")
        print(f"Mean IoU: {iou.mean():.4f}")
        print(f"Dice below 0.50: {(dice < 0.50).sum()}")
        print(f"Dice below 0.70: {(dice < 0.70).sum()}")
        print(f"Empty predictions: {(pred == 0).sum()}")

    differences = np.array([
        r["continued_dice"] - r["baseline_dice"]
        for r in rows
    ])

    print("\nPER-CASE COMPARISON")
    print("Improved cases:", (differences > 1e-6).sum())
    print("Unchanged cases:", (np.abs(differences) <= 1e-6).sum())
    print("Regressed cases:", (differences < -1e-6).sum())
    print("Mean per-case Dice change:", f"{differences.mean():.4f}")

    print("\nSaved comparison:", OUTPUT)


if __name__ == "__main__":
    main()
