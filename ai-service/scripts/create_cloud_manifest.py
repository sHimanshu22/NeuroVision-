import csv
from pathlib import Path

ROOT = Path(r"S:\NeuroVision\cloud_transfer")
SOURCE = ROOT / "data" / "segmentation_manifest.csv"
OUTPUT = ROOT / "data" / "segmentation_manifest_cloud.csv"

with SOURCE.open("r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    row["image_path"] = (
        f"dataset/segmentation_task/{row['split']}/images/{row['filename']}"
    )

    mask_name = Path(row["mask_path"]).name
    row["mask_path"] = (
        f"dataset/segmentation_task/{row['split']}/masks/{mask_name}"
    )

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Rows:", len(rows))
print("Cloud manifest:", OUTPUT)
print("Sample rows:")
for row in rows[:2]:
    print(row)

excluded = [
    row for row in rows
    if row["split"] == "train"
    and row["included_for_training"].strip().lower() != "true"
]
print("Excluded training rows:", len(excluded))
for row in excluded:
    print(row["filename"], "|", row["exclusion_reason"])
