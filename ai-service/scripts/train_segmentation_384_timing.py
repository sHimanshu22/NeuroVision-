import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

from segmentation_dataset import BrainSegmentationDataset
from unet_model import UNet


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"

OUTPUT_DIR = ROOT / "outputs" / "segmentation" / "timing_384"
CHECKPOINT_PATH = OUTPUT_DIR / "timing_checkpoint.pt"
HISTORY_PATH = OUTPUT_DIR / "timing_history.csv"

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_loss(logits, targets, smooth=1.0):
    probabilities = torch.sigmoid(logits.float())
    probabilities = probabilities.flatten(start_dim=1)
    targets = targets.float().flatten(start_dim=1)

    intersection = (probabilities * targets).sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        probabilities.sum(dim=1) + targets.sum(dim=1) + smooth
    )

    return 1.0 - dice.mean()


@torch.no_grad()
def calculate_metrics(logits, targets, threshold=0.5, smooth=1.0):
    predictions = (torch.sigmoid(logits.float()) >= threshold).float()

    predictions = predictions.flatten(start_dim=1)
    targets = targets.float().flatten(start_dim=1)

    intersection = (predictions * targets).sum(dim=1)
    prediction_sum = predictions.sum(dim=1)
    target_sum = targets.sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        prediction_sum + target_sum + smooth
    )

    union = prediction_sum + target_sum - intersection
    iou = (intersection + smooth) / (union + smooth)

    return dice.mean().item(), iou.mean().item()


class CheckpointedUNet(UNet):
    """U-Net that recomputes convolution-block activations during training."""

    def _run_block(self, block, x):
        if self.training and torch.is_grad_enabled():
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x):
        e1 = self._run_block(self.enc1, x)
        e2 = self._run_block(self.enc2, self.pool(e1))
        e3 = self._run_block(self.enc3, self.pool(e2))
        e4 = self._run_block(self.enc4, self.pool(e3))

        b = self._run_block(self.bottleneck, self.pool(e4))

        d4 = self.up4(b)
        d4 = self._run_block(
            self.dec4, torch.cat([d4, e4], dim=1)
        )

        d3 = self.up3(d4)
        d3 = self._run_block(
            self.dec3, torch.cat([d3, e3], dim=1)
        )

        d2 = self.up2(d3)
        d2 = self._run_block(
            self.dec2, torch.cat([d2, e2], dim=1)
        )

        d1 = self.up1(d2)
        d1 = self._run_block(
            self.dec1, torch.cat([d1, e1], dim=1)
        )

        return self.output(d1)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    max_batches=None,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    processed = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True).float()

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            # FP32 is deliberate: FP16 produced NaN loss on this GPU.
            logits = model(images)
            bce = criterion(logits, masks)
            d_loss = dice_loss(logits, masks)
            loss = 0.5 * bce + 0.5 * d_loss

            if training:
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss at batch {batch_index}: "
                        f"{loss.item()}"
                    )

                loss.backward()
                optimizer.step()

        dice, iou = calculate_metrics(logits.detach(), masks)

        batch_size = images.size(0)
        total_loss += loss.detach().item() * batch_size
        total_dice += dice * batch_size
        total_iou += iou * batch_size
        processed += batch_size

        del images, masks, logits, loss, bce, d_loss

    if processed == 0:
        raise RuntimeError("No batches were processed.")

    return {
        "loss": total_loss / processed,
        "dice": total_dice / processed,
        "iou": total_iou / processed,
        "samples": processed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")

    set_seed(args.seed)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Mixed precision: disabled (FP32 for numerical stability)")
    print("Activation checkpointing: enabled during training")

    image_size = 384

    train_dataset = BrainSegmentationDataset(
        manifest_path=MANIFEST,
        split="train",
        image_size=image_size,
        mask_threshold=128,
        validation_fraction=0.10,
        seed=args.seed,
    )

    val_dataset = BrainSegmentationDataset(
        manifest_path=MANIFEST,
        split="validation",
        image_size=image_size,
        mask_threshold=128,
        validation_fraction=0.10,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = CheckpointedUNet(
        in_channels=1,
        out_channels=1,
        base_channels=16,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_val_dice = -1.0
    epochs_without_improvement = 0

    print("Image size:", image_size)
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.learning_rate)
    print("Output directory:", OUTPUT_DIR)

    if args.smoke_test:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            max_batches=3,
        )

        print("Smoke-test metrics:", metrics)

        if device.type == "cuda":
            torch.cuda.synchronize()
            print(
                "Peak GPU memory allocated:",
                round(
                    torch.cuda.max_memory_allocated(device) / (1024 ** 2),
                    2,
                ),
                "MiB",
            )

        print("384x384 FP32 checkpointing smoke test PASSED.")
        return

    with HISTORY_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        fieldnames = [
            "epoch",
            "train_loss",
            "train_dice",
            "train_iou",
            "val_loss",
            "val_dice",
            "val_iou",
            "learning_rate",
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
            )

            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
            )

            scheduler.step(val_metrics["dice"])
            current_lr = optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "train_iou": train_metrics["iou"],
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "learning_rate": current_lr,
            }

            writer.writerow(row)
            csv_file.flush()

            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train loss={train_metrics['loss']:.4f} | "
                f"train Dice={train_metrics['dice']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} | "
                f"val Dice={val_metrics['dice']:.4f} | "
                f"val IoU={val_metrics['iou']:.4f} | "
                f"lr={current_lr:.2e}"
            )

            if device.type == "cuda":
                print(
                    "  Peak allocated GPU memory:",
                    round(
                        torch.cuda.max_memory_allocated(device)
                        / (1024 ** 2),
                        2,
                    ),
                    "MiB",
                )

            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                epochs_without_improvement = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_dice": best_val_dice,
                        "val_iou": val_metrics["iou"],
                        "config": vars(args),
                        "image_size": image_size,
                        "mask_threshold": 128,
                        "seed": args.seed,
                        "amp": False,
                        "activation_checkpointing": True,
                    },
                    CHECKPOINT_PATH,
                )

                print("  Saved best 384x384 checkpoint.")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping after {epoch} epochs. "
                    f"Best validation Dice: {best_val_dice:.4f}"
                )
                break

    print("\n384x384 training finished.")
    print("Best validation Dice:", round(best_val_dice, 4))
    print("Checkpoint:", CHECKPOINT_PATH)
    print("History:", HISTORY_PATH)


if __name__ == "__main__":
    main()
