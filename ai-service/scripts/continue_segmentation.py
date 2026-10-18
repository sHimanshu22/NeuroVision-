
import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from segmentation_dataset import BrainSegmentationDataset
from unet_model import UNet
from train_segmentation import run_epoch, set_seed


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"
BASELINE = ROOT / "outputs" / "segmentation" / "best_unet.pt"

EXPERIMENT_DIR = ROOT / "outputs" / "segmentation" / "continuation_lr1e4"
CHECKPOINT_PATH = EXPERIMENT_DIR / "best_continued_unet.pt"
HISTORY_PATH = EXPERIMENT_DIR / "continuation_history.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
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

    train_dataset = BrainSegmentationDataset(
        manifest_path=MANIFEST,
        split="train",
        image_size=256,
        mask_threshold=128,
        validation_fraction=0.10,
        seed=args.seed,
    )

    val_dataset = BrainSegmentationDataset(
        manifest_path=MANIFEST,
        split="validation",
        image_size=256,
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

    checkpoint = torch.load(
        BASELINE,
        map_location=device,
        weights_only=False,
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=16,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    # Start a fresh, lower learning-rate optimization phase.
    # We deliberately do not restore the old optimizer state.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    criterion = nn.BCEWithLogitsLoss()

    baseline_dice = float(checkpoint["best_val_dice"])
    best_val_dice = baseline_dice
    best_val_iou = float(checkpoint["val_iou"])
    epochs_without_improvement = 0

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    print("Baseline checkpoint epoch:", checkpoint["epoch"])
    print("Baseline validation Dice:", f"{baseline_dice:.4f}")
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Learning rate:", args.learning_rate)
    print("Experiment output:", EXPERIMENT_DIR)

    with HISTORY_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        fieldnames = [
            "continuation_epoch",
            "source_epoch",
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
                "continuation_epoch": epoch,
                "source_epoch": int(checkpoint["epoch"]) + epoch,
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
                f"Continuation {epoch:02d}/{args.epochs} | "
                f"train loss={train_metrics['loss']:.4f} | "
                f"train Dice={train_metrics['dice']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} | "
                f"val Dice={val_metrics['dice']:.4f} | "
                f"val IoU={val_metrics['iou']:.4f} | "
                f"lr={current_lr:.2e}"
            )

            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                best_val_iou = val_metrics["iou"]
                epochs_without_improvement = 0

                torch.save(
                    {
                        "epoch": int(checkpoint["epoch"]) + epoch,
                        "continuation_epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_dice": best_val_dice,
                        "val_iou": best_val_iou,
                        "config": vars(args),
                        "image_size": 256,
                        "mask_threshold": 128,
                        "seed": args.seed,
                        "baseline_checkpoint": str(BASELINE),
                    },
                    CHECKPOINT_PATH,
                )

                print("  Saved improved continuation checkpoint.")

            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print("Early stopping: validation Dice did not improve.")
                break

    print("\nContinuation finished.")
    print("Original baseline Dice:", f"{baseline_dice:.4f}")
    print("Best continuation Dice:", f"{best_val_dice:.4f}")
    print("Best continuation IoU:", f"{best_val_iou:.4f}")
    print("Continuation checkpoint:", CHECKPOINT_PATH)
    print("Continuation history:", HISTORY_PATH)

    if best_val_dice <= baseline_dice:
        print(
            "No improvement over the baseline. "
            "Keep using the original checkpoint."
        )


if __name__ == "__main__":
    main()
