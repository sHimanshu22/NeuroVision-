
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from segmentation_dataset import BrainSegmentationDataset
from unet_model import UNet
from train_segmentation import run_epoch


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"


def main():
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    train_dataset = BrainSegmentationDataset(
        manifest_path=MANIFEST,
        split="train",
        image_size=384,
        mask_threshold=128,
        validation_fraction=0.10,
        seed=42,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=16,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    metrics = run_epoch(
        model,
        loader,
        criterion,
        device,
        optimizer=optimizer,
        max_batches=3,
    )

    print("Device:", device)
    print("Tested input shape:", (1, 1, 384, 384))
    print("Smoke-test metrics:", metrics)
    print("384x384 training pipeline smoke test PASSED.")


if __name__ == "__main__":
    main()
