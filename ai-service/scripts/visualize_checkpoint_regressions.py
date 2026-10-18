
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from segmentation_dataset import BrainSegmentationDataset
from unet_model import UNet


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"
COMPARISON = ROOT / "outputs" / "segmentation" / "checkpoint_comparison.csv"

BASELINE = ROOT / "outputs" / "segmentation" / "best_unet.pt"
CONTINUED = (
    ROOT / "outputs" / "segmentation" / "continuation_lr1e4"
    / "best_continued_unet.pt"
)

OUTPUT = (
    ROOT / "outputs" / "segmentation"
    / "checkpoint_regressions.png"
)

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


def load_model(path):
    checkpoint = torch.load(
        path, map_location=DEVICE, weights_only=False
    )

    model = UNet(in_channels=1, out_channels=1, base_channels=16)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    return model


def main():
    with COMPARISON.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows.sort(
        key=lambda row:
        float(row["continued_dice"]) - float(row["baseline_dice"])
    )
    worst = rows[:10]

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

    baseline = load_model(BASELINE)
    continued = load_model(CONTINUED)

    fig, axes = plt.subplots(10, 4, figsize=(14, 30))

    with torch.inference_mode():
        for i, row in enumerate(worst):
            sample = dataset[index_by_name[row["filename"]]]
            image = sample["image"].unsqueeze(0).to(DEVICE)
            target = sample["mask"][0].numpy()

            baseline_pred = (
                torch.sigmoid(baseline(image))[0, 0].cpu().numpy()
                >= 0.5
            )
            continued_pred = (
                torch.sigmoid(continued(image))[0, 0].cpu().numpy()
                >= 0.5
            )

            axes[i, 0].imshow(sample["image"][0].numpy(), cmap="gray")
            axes[i, 0].set_title(row["filename"], fontsize=7)

            axes[i, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title("Ground truth")

            axes[i, 2].imshow(baseline_pred, cmap="gray", vmin=0, vmax=1)
            axes[i, 2].set_title(
                f"Baseline Dice {float(row['baseline_dice']):.3f}"
            )

            axes[i, 3].imshow(continued_pred, cmap="gray", vmin=0, vmax=1)
            axes[i, 3].set_title(
                f"Continued Dice {float(row['continued_dice']):.3f}"
            )

            for ax in axes[i]:
                ax.axis("off")

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
