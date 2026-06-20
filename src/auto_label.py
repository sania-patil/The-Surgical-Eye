"""
Auto-labeling script using Grounding DINO
Generates YOLO format labels for training images
Run this first - it will run in background while you build other components
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from groundingdino.util.inference import load_model, load_image, predict

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_DIR       = Path("D:/TDS/outputs/enhanced_frames")
LABELS_DIR      = Path("D:/TDS/dataset/labels")
CONFIG_PATH     = "D:/TDS/groundingdino_config.py"
WEIGHTS_PATH    = "D:/TDS/groundingdino_swint_ogc.pth"
TEXT_PROMPT     = "kidney stone . calculus . renal stone"
BOX_THRESHOLD   = 0.30
TEXT_THRESHOLD  = 0.25
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

def box_to_yolo(box, img_w, img_h):
    """Convert [cx, cy, w, h] normalized box to YOLO format."""
    cx, cy, w, h = box
    return cx, cy, w, h  # already normalized by Grounding DINO

def run_auto_labeling():
    print(f"Using device: {DEVICE}")
    print("Loading Grounding DINO model...")

    model = load_model(CONFIG_PATH, WEIGHTS_PATH, device=DEVICE)
    print("Model loaded successfully!")

    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    image_files = list(TRAIN_DIR.glob("*.jpg")) + list(TRAIN_DIR.glob("*.png"))
    print(f"Found {len(image_files)} training images")

    labeled_count = 0
    skipped_count = 0

    for img_path in tqdm(image_files, desc="Auto-labeling"):
        label_path = LABELS_DIR / (img_path.stem + ".txt")

        # skip if already labeled
        if label_path.exists():
            skipped_count += 1
            continue

        try:
            image_source, image = load_image(str(img_path))
            img_h, img_w = image_source.shape[:2]

            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=DEVICE
            )

            lines = []
            if len(boxes) > 0:
                for box, logit in zip(boxes.tolist(), logits.tolist()):
                    cx, cy, w, h = box
                    # class 0 = kidney_stone
                    lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                labeled_count += 1

            # write label file (empty if no detection = background image)
            with open(label_path, "w") as f:
                f.write("\n".join(lines))

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue

    print(f"\nAuto-labeling complete!")
    print(f"  Labeled with detections : {labeled_count}")
    print(f"  Skipped (already done)  : {skipped_count}")
    print(f"  Total processed         : {labeled_count + skipped_count}")
    print(f"  Labels saved to         : {LABELS_DIR}")

if __name__ == "__main__":
    run_auto_labeling()
