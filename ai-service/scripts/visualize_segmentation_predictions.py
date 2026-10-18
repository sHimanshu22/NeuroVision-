from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from segmentation_dataset import BrainSegmentationDataset
from unet_model import UNet


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"

CHECKPOINT = (
    ROOT / "outputs" / "segmentation"
    / "continuation_lr1e4" / "best_continued_unet.pt"
)
METRICS_CSV = (
    ROOT / "outputs" / "segmentation" / "validation_case_metrics.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "segmentation" / "worst_case_previews"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 10
CASES_PER_IMAGE = 2
FIGURE_DPI = 90


def calculate_dice(prediction, target, smooth=1.0):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    return (
        2.0 * intersection + smooth
    ) / (
        prediction.sum() + target.sum() + smooth
    )


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    if not METRICS_CSV.is_file():
        raise FileNotFoundError(
            f"Validation metrics not found: {METRICS_CSV}. "
            "Run evaluate_segmentation_validation.py first."
        )

    dataset = BrainSegmentationDataset(
        manifest_path=MANIFEST,
        split="validation",
        image_size=256,
        mask_threshold=128,
        validation_fraction=0.10,
        seed=42,
    )

    with METRICS_CSV.open("r", newline="", encoding="utf-8") as f:
        metric_rows = list(csv.DictReader(f))
    if not metric_rows:
        raise RuntimeError(f"No metric rows found in {METRICS_CSV}")

    worst_rows = sorted(metric_rows, key=lambda row: float(row["dice"]))[
        :NUM_SAMPLES
    ]

    dataset_indices = {
        dataset[i]["filename"]: i for i in range(len(dataset))
    }
    selected = [
        (row, dataset_indices[row["filename"]])
        for row in worst_rows
        if row["filename"] in dataset_indices
    ]
    if not selected:
        raise RuntimeError("No matching validation cases found.")

    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=False
    )
    model = UNet(
        in_channels=1, out_channels=1, base_channels=16
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dice_scores = []

    # Render only two cases per figure to keep RAM usage low.
    for start in range(0, len(selected), CASES_PER_IMAGE):
        batch = selected[start:start + CASES_PER_IMAGE]
        fig, axes = plt.subplots(
            len(batch), 4,
            figsize=(10, 2.35 * len(batch)),
            squeeze=False,
        )

        for local_i, (metric_row, dataset_index) in enumerate(batch):
            sample = dataset[dataset_index]
            image = sample["image"].unsqueeze(0).to(DEVICE)

            target = sample["mask"][0].numpy().astype(bool)
            with torch.inference_mode():
                logits = model(image)
                probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
            prediction = probability >= 0.5

            dice = calculate_dice(prediction, target)
            dice_scores.append(dice)
            mri = sample["image"][0].numpy()

            gt_overlay = np.zeros((*target.shape, 3), dtype=np.float32)
            gt_overlay[..., 1] = target
            pred_overlay = np.zeros(
                (*prediction.shape, 3), dtype=np.float32
            )
            pred_overlay[..., 0] = prediction

            axes[local_i, 0].imshow(mri, cmap="gray")
            axes[local_i, 0].set_title("MRI", fontsize=8)

            axes[local_i, 1].imshow(mri, cmap="gray")
            axes[local_i, 1].imshow(gt_overlay, alpha=0.55)
            axes[local_i, 1].set_title("Ground truth (green)", fontsize=8)

            axes[local_i, 2].imshow(mri, cmap="gray")
            axes[local_i, 2].imshow(pred_overlay, alpha=0.55)
            axes[local_i, 2].set_title("Prediction (red)", fontsize=8)

            axes[local_i, 3].imshow(target, cmap="gray", vmin=0, vmax=1)
            axes[local_i, 3].imshow(
                prediction, cmap="autumn", alpha=0.45, vmin=0, vmax=1
            )
            axes[local_i, 3].set_title(f"Overlay | Dice {dice:.3f}", fontsize=8)

            for ax in axes[local_i]:
                ax.axis("off")

            print(
                f"{sample['filename']} | Dice={dice:.4f} | "
                f"GT pixels={int(target.sum())} | "
                f"Predicted pixels={int(prediction.sum())}"
            )

        fig.suptitle(
            f"Worst validation predictions ({start + 1}-{start + len(batch)})",
            fontsize=10,
        )
        # Avoid tight_layout, which triggered a large memory allocation.
        fig.subplots_adjust(
            left=0.02, right=0.98, top=0.88, bottom=0.03,
            wspace=0.08, hspace=0.18,
        )

        output_path = OUTPUT_DIR / f"worst_cases_{start + 1:02d}_{start + len(batch):02d}.png"
        fig.savefig(
            output_path,
            dpi=FIGURE_DPI,
            bbox_inches=None,
            pil_kwargs={"compress_level": 6},
        )
        plt.close(fig)
        print("Saved:", output_path)

    print("\nCheckpoint epoch:", checkpoint.get("epoch", "unknown"))
    print(
        "Best validation Dice recorded in checkpoint:",
        checkpoint.get("best_val_dice", "unknown"),
    )
    print("Mean Dice of displayed cases:", round(float(np.mean(dice_scores)), 4))
    print("All preview images saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
