
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from segmentation_dataset import BrainSegmentationDataset
from unet_model import UNet


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Check your PyTorch/GPU setup."
    )

device = torch.device("cuda:0")
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

print("GPU:", torch.cuda.get_device_name(0))

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
    batch_size=2,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

batch = next(iter(loader))
images = batch["image"].to(device)
masks = batch["mask"].to(device)

model = UNet(
    in_channels=1,
    out_channels=1,
    base_channels=16,
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

parameter_count = sum(p.numel() for p in model.parameters())

print("Model parameters:", f"{parameter_count:,}")
print("Input shape:", tuple(images.shape))
print("Target shape:", tuple(masks.shape))

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

model.train()
optimizer.zero_grad(set_to_none=True)

logits = model(images)

assert logits.shape == masks.shape, (
    f"Output {logits.shape} does not match target {masks.shape}"
)

loss = criterion(logits, masks)

if not torch.isfinite(loss):
    raise RuntimeError(f"Non-finite loss: {loss.item()}")

loss.backward()

gradient_count = sum(
    1 for p in model.parameters()
    if p.grad is not None
)

if gradient_count == 0:
    raise RuntimeError("No gradients were calculated.")

optimizer.step()
torch.cuda.synchronize()

peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

print("\n--- U-NET GPU SMOKE TEST ---")
print("Output shape:", tuple(logits.shape))
print("Loss:", round(loss.item(), 6))
print("Parameters with gradients:", gradient_count)
print("Peak allocated GPU memory:", round(peak_mb, 2), "MB")
print("GPU allocated after step:",
      round(torch.cuda.memory_allocated() / (1024 ** 2), 2), "MB")
print("\nU-Net forward/backward test PASSED.")
