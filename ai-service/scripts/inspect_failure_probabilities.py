
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from segmentation_dataset import BrainSegmentationDataset
from unet_model import UNet


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"
METRICS = ROOT / "outputs" / "segmentation" / "validation_case_metrics.csv"
CHECKPOINT = ROOT / "outputs" / "segmentation" / "best_unet.pt"
OUTPUT = ROOT / "outputs" / "segmentation" / "failure_probabilities.png"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    with METRICS.open("r", newline="", encoding="utf-8") as f:
        worst = sorted(
            csv.DictReader(f),
            key=lambda row: float(row["dice"])
        )[:6]

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

    fig, axes = plt.subplots(6, 4, figsize=(14, 24), squeeze=False)

    with torch.inference_mode():
        for i, record in enumerate(worst):
            filename = record["filename"]
            sample = dataset[index_by_name[filename]]

            image = sample["image"].unsqueeze(0).to(DEVICE)
            target = sample["mask"][0].numpy().astype(bool)

            probability = torch.sigmoid(model(image))[0, 0].cpu().numpy()
            prediction = probability >= 0.5

            axes[i, 0].imshow(sample["image"][0].numpy(), cmap="gray")
            axes[i, 0].set_title(filename, fontsize=8)

            axes[i, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title("Ground truth")

            im = axes[i, 2].imshow(
                probability, cmap="viridis", vmin=0, vmax=1
            )
            axes[i, 2].set_title(
                f"Probability map\nmax={probability.max():.3f}"
            )

            axes[i, 3].imshow(prediction, cmap="Reds", vmin=0, vmax=1)
            axes[i, 3].set_title(
                f"Threshold 0.50\npred pixels={int(prediction.sum())}"
            )

            print(
                f"{filename} | "
                f"min={probability.min():.4f} | "
                f"mean={probability.mean():.4f} | "
                f"max={probability.max():.4f} | "
                f"GT pixels={int(target.sum())} | "
                f"predicted pixels={int(prediction.sum())}"
            )

            for ax in axes[i]:
                ax.axis("off")

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
