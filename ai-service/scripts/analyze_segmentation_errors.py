
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "outputs" / "segmentation" / "validation_case_metrics.csv"

with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    row["dice"] = float(row["dice"])
    row["iou"] = float(row["iou"])
    row["gt_pixels"] = int(row["ground_truth_foreground_pixels"])
    row["pred_pixels"] = int(row["predicted_foreground_pixels"])

groups = [
    ("Very small: < 500 pixels", lambda r: r["gt_pixels"] < 500),
    ("Small: 500–999 pixels", lambda r: 500 <= r["gt_pixels"] < 1000),
    ("Medium: 1000–2999 pixels", lambda r: 1000 <= r["gt_pixels"] < 3000),
    ("Large: >= 3000 pixels", lambda r: r["gt_pixels"] >= 3000),
]

print("Validation error analysis at prediction threshold 0.50")
print("Total samples:", len(rows))

for label, condition in groups:
    selected = [r for r in rows if condition(r)]

    if not selected:
        print(f"\n{label}: no samples")
        continue

    dice = np.array([r["dice"] for r in selected])
    iou = np.array([r["iou"] for r in selected])

    empty_predictions = sum(r["pred_pixels"] == 0 for r in selected)

    print(f"\n{label}")
    print("  Samples:", len(selected))
    print("  Mean Dice:", round(float(dice.mean()), 4))
    print("  Median Dice:", round(float(np.median(dice)), 4))
    print("  Mean IoU:", round(float(iou.mean()), 4))
    print("  Empty predictions:", empty_predictions)

print("\nWorst 10 cases:")
for row in sorted(rows, key=lambda r: r["dice"])[:10]:
    print(
        f"{row['filename']} | "
        f"GT pixels={row['gt_pixels']} | "
        f"Predicted pixels={row['pred_pixels']} | "
        f"Dice={row['dice']:.4f}"
    )
