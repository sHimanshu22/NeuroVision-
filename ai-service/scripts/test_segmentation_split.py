
from pathlib import Path

from segmentation_dataset import BrainSegmentationDataset


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"

common = {
    "manifest_path": MANIFEST,
    "image_size": 256,
    "mask_threshold": 128,
    "validation_fraction": 0.10,
    "seed": 42,
}

train = BrainSegmentationDataset(split="train", **common)
validation = BrainSegmentationDataset(split="validation", **common)
test = BrainSegmentationDataset(split="test", **common)

train_names = {s["filename"] for s in train.samples}
validation_names = {s["filename"] for s in validation.samples}
test_names = {s["filename"] for s in test.samples}

assert train_names.isdisjoint(validation_names), (
    "Training and validation filenames overlap!"
)
assert train_names.isdisjoint(test_names), (
    "Training and test filenames overlap!"
)
assert validation_names.isdisjoint(test_names), (
    "Validation and test filenames overlap!"
)

assert len(train) + len(validation) == 3924, (
    "Eligible training records were lost or duplicated."
)
assert len(test) == 860, (
    "Official test set count changed."
)

print("\n--- SPLIT VERIFICATION PASSED ---")
print("Training samples:", len(train))
print("Validation samples:", len(validation))
print("Official test samples:", len(test))
print("Train/validation overlap:", len(train_names & validation_names))
print("Train/test overlap:", len(train_names & test_names))
print("Validation/test overlap:", len(validation_names & test_names))

sample = train[0]
print("\nSample image shape:", tuple(sample["image"].shape))
print("Sample mask shape:", tuple(sample["mask"].shape))
print("Image dtype:", sample["image"].dtype)
print("Mask dtype:", sample["mask"].dtype)
print("Mask values:", sample["mask"].unique().tolist())
print("Image range:", sample["image"].min().item(),
      sample["image"].max().item())
print("Mask foreground fraction:",
      sample["mask"].mean().item())
