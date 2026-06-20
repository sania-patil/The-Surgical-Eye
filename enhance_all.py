"""
Enhance all training images using CLAHE + denoising
Saves enhanced images to outputs/enhanced_frames/
"""

import sys
sys.path.insert(0, "C:/Users/Cctech/TDS/src")

import cv2
from pathlib import Path
from tqdm import tqdm
from enhance import enhance_frame, get_visibility_score

TRAIN_DIR   = Path("C:/Users/Cctech/TDS/dataset/train")
OUTPUT_DIR  = Path("D:/TDS/outputs/enhanced_frames")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_files = list(TRAIN_DIR.glob("*.jpg"))
print(f"Found {len(image_files)} training images")

visibility_log = []

for img_path in tqdm(image_files, desc="Enhancing"):
    frame = cv2.imread(str(img_path))
    if frame is None:
        continue

    enhanced = enhance_frame(frame)
    vis      = get_visibility_score(enhanced)
    visibility_log.append({"file": img_path.name, **vis})

    # Save enhanced image with same name
    out_path = OUTPUT_DIR / img_path.name
    cv2.imwrite(str(out_path), enhanced)

# Summary
good  = sum(1 for v in visibility_log if v["label"] == "Good")
poor  = sum(1 for v in visibility_log if v["label"] == "Poor")
vbad  = sum(1 for v in visibility_log if v["label"] == "Very Poor")

print(f"\nEnhancement complete!")
print(f"  Total   : {len(visibility_log)}")
print(f"  Good    : {good}")
print(f"  Poor    : {poor}")
print(f"  Very Poor: {vbad}")
print(f"  Saved to: {OUTPUT_DIR}")
