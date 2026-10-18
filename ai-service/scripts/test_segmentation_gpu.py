
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from segmentation_dataset import BrainSegmentationDataset


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.cuda.reset_peak_memory_stats()

dataset = BrainSegmentationDataset(
    manifest_path=MANIFEST,
    split="train",
    image_size=256,
    mask_threshold=128,
    validation_fraction=0.10,
    seed=42,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

batch = next(iter(loader))

images = batch["image"].to(device)
masks = batch["mask"].to(device)

print("\n--- GPU BATCH TEST ---")
print("Image batch shape:", tuple(images.shape))
print("Mask batch shape:", tuple(masks.shape))
print("Image dtype:", images.dtype)
print("Mask dtype:", masks.dtype)
print("Image device:", images.device)
print("Mask device:", masks.device)
print("Mask values:", torch.unique(masks).cpu().tolist())
print("Image range:", images.min().item(), images.max().item())
print("Mask foreground fraction:", masks.mean().item())

if device.type == "cuda":
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print("Peak allocated GPU memory:", round(peak_mb, 2), "MB")

assert images.shape == (4, 1, 256, 256)
assert masks.shape == (4, 1, 256, 256)
assert images.device.type == device.type
assert masks.device.type == device.type

if device.type == "cuda":
    assert images.device.index == torch.cuda.current_device()
    assert masks.device.index == torch.cuda.current_device()
    
assert set(torch.unique(masks).cpu().tolist()).issubset({0.0, 1.0})

print("\nGPU batch smoke test PASSED.")
