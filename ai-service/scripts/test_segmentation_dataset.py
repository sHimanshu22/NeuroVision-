
from pathlib import Path

import torch
from segmentation_dataset import BrainSegmentationDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "segmentation_manifest.csv"


def main():
    train_dataset = BrainSegmentationDataset(
        manifest_path=MANIFEST_PATH,
        split="train",
        image_size=256,
        mask_threshold=128,
    )

    test_dataset = BrainSegmentationDataset(
        manifest_path=MANIFEST_PATH,
        split="test",
        image_size=256,
        mask_threshold=128,
    )

    print("\nDataset counts")
    print("Training:", len(train_dataset))
    print("Official test:", len(test_dataset))

    sample = train_dataset[0]
    image = sample["image"]
    mask = sample["mask"]

    print("\nFirst training sample")
    print("Filename:", sample["filename"])
    print("Image shape:", tuple(image.shape))
    print("Mask shape:", tuple(mask.shape))
    print("Image dtype:", image.dtype)
    print("Mask dtype:", mask.dtype)
    print("Image range:", image.min().item(), image.max().item())
    print("Mask values:", torch.unique(mask).tolist())
    print("Mask foreground fraction:", mask.mean().item())

    assert image.shape == (1, 256, 256)
    assert mask.shape == (1, 256, 256)
    assert image.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert torch.isfinite(image).all()
    assert torch.isfinite(mask).all()
    assert image.min() >= 0 and image.max() <= 1
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})

    assert len(train_dataset) == 3924
    assert len(test_dataset) == 860

    print("\nPASS: Dataset loader checks succeeded.")


if __name__ == "__main__":
    main()
