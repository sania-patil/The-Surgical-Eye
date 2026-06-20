"""
Prepare YOLO dataset structure
Copies matched image+label pairs to proper YOLO format
"""

import shutil
from pathlib import Path
from tqdm import tqdm

ENHANCED_DIR = Path("D:/TDS/outputs/enhanced_frames")
LABELS_DIR   = Path("D:/TDS/dataset/labels")
YOLO_DIR     = Path("D:/TDS/dataset/yolo")

# Create YOLO structure
(YOLO_DIR / "images" / "train").mkdir(parents=True, exist_ok=True)
(YOLO_DIR / "labels" / "train").mkdir(parents=True, exist_ok=True)

label_files = list(LABELS_DIR.glob("*.txt"))
matched = 0

for label_path in tqdm(label_files, desc="Preparing dataset"):
    # Find matching image
    img_path = ENHANCED_DIR / (label_path.stem + ".jpg")
    if not img_path.exists():
        continue

    # Only copy if label has content (has detections)
    if label_path.stat().st_size == 0:
        continue

    shutil.copy(img_path,   YOLO_DIR / "images" / "train" / img_path.name)
    shutil.copy(label_path, YOLO_DIR / "labels" / "train" / label_path.name)
    matched += 1

print(f"\nDataset ready: {matched} image-label pairs")
print(f"Saved to: {YOLO_DIR}")
