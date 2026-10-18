
import csv
from pathlib import Path

import torch

from segmentation_dataset import BrainSegmentationDataset
from unet_model import UNet


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"
METRICS = ROOT / "outputs" / "segmentation" / "validation_case_metrics.csv"
CHECKPOINT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "continuation_lr1e4"
    / "best_continued_unet.pt"
)

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
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    model_a = UNet(in_channels=1, out_channels=1, base_channels=16).to(DEVICE)
    model_b = UNet(in_channels=1, out_channels=1, base_channels=16).to(DEVICE)

    model_a.load_state_dict(checkpoint["model_state_dict"])
    model_b.load_state_dict(checkpoint["model_state_dict"])

    model_a.eval()
    model_b.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Checking reproducibility on six difficult validation samples.\n")

    with torch.inference_mode():
        for record in worst:
            filename = record["filename"]
            sample = dataset[index_by_name[filename]]
            image = sample["image"].unsqueeze(0).to(DEVICE)

            logits_a = model_a(image)
            logits_b = model_b(image)

            prob_a = torch.sigmoid(logits_a)
            prob_b = torch.sigmoid(logits_b)

            max_difference = (prob_a - prob_b).abs().max().item()
            pred_a = prob_a >= 0.5
            pred_b = prob_b >= 0.5

            same_prediction = torch.equal(pred_a, pred_b)

            print(filename)
            print("  Maximum probability difference:", max_difference)
            print("  Binary predictions identical:", same_prediction)
            print("  Maximum predicted probability:", prob_a.max().item())
            print("  Predicted foreground pixels:", pred_a.sum().item())

            if not same_prediction:
                raise RuntimeError(
                    f"Predictions differ between identical checkpoint loads: {filename}"
                )

    print("\nCheckpoint reproducibility test PASSED.")


if __name__ == "__main__":
    main()
