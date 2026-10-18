
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "outputs" / "segmentation" / "validation_case_metrics.csv"

groups = defaultdict(list)

with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        parts = Path(row["filename"]).stem.split("_")

        # Example: brisc2025_train_01291_me_ax_t1
        plane = parts[-2]

        groups[plane].append({
            "filename": row["filename"],
            "dice": float(row["dice"]),
            "iou": float(row["iou"]),
            "gt_pixels": int(row["ground_truth_foreground_pixels"]),
            "pred_pixels": int(row["predicted_foreground_pixels"]),
        })

print("Validation performance by MRI plane")

for plane, rows in sorted(groups.items()):
    dice = np.array([r["dice"] for r in rows])
    iou = np.array([r["iou"] for r in rows])

    print(f"\nPlane: {plane}")
    print("  Samples:", len(rows))
    print("  Mean Dice:", round(float(dice.mean()), 4))
    print("  Median Dice:", round(float(np.median(dice)), 4))
    print("  Mean IoU:", round(float(iou.mean()), 4))
    print("  Dice < 0.50:", int((dice < 0.50).sum()))
    print("  Empty predictions:", sum(r["pred_pixels"] == 0 for r in rows))

    print("  Worst 3:")
    for r in sorted(rows, key=lambda x: x["dice"])[:3]:
        print(
            f"    {r['filename']} | Dice={r['dice']:.4f} | "
            f"GT={r['gt_pixels']} | Pred={r['pred_pixels']}"
        )
