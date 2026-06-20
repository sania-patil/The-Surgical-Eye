"""
Test Grounding DINO detection on one training image
"""

import sys
sys.path.insert(0, "C:/Users/Cctech/TDS/src")

import cv2
import torch
from pathlib import Path
from enhance import enhance_frame

# Load image
img_path = "C:/Users/Cctech/TDS/dataset/train/002-cropped_mp4-0000_jpg.rf.7S4lA0Dwr1u8u50EZ919.jpg"
frame = cv2.imread(img_path)
enhanced = enhance_frame(frame)

print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print("Loading Grounding DINO...")

from groundingdino.util.inference import load_model, load_image, predict
from PIL import Image
import tempfile, os, numpy as np

CONFIG_PATH  = "D:/TDS/groundingdino_config.py"
WEIGHTS_PATH = "D:/TDS/groundingdino_swint_ogc.pth"
TEXT_PROMPT  = "dark kidney stone . black calculus . stone"
BOX_THRESHOLD  = 0.25
TEXT_THRESHOLD = 0.20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = load_model(CONFIG_PATH, WEIGHTS_PATH, device=DEVICE)
print("Model loaded!")

# Save enhanced image to temp file for DINO
rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
pil = Image.fromarray(rgb)

with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    tmp_path = tmp.name
    pil.save(tmp_path)

image_source, image = load_image(tmp_path)
os.unlink(tmp_path)

h, w = enhanced.shape[:2]

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    device=DEVICE
)

# Apply NMS to remove duplicate boxes
from torchvision.ops import nms
import torch as t
if len(boxes) > 1:
    # convert cx,cy,w,h to x1,y1,x2,y2 for NMS
    b = boxes.clone()
    x1 = (b[:,0] - b[:,2]/2) * w
    y1 = (b[:,1] - b[:,3]/2) * h
    x2 = (b[:,0] + b[:,2]/2) * w
    y2 = (b[:,1] + b[:,3]/2) * h
    xyxy = t.stack([x1,y1,x2,y2], dim=1)
    keep = nms(xyxy, t.tensor(logits), iou_threshold=0.5)
    boxes  = boxes[keep]
    logits = logits[keep]
    phrases = [phrases[i] for i in keep.tolist()]

h, w = enhanced.shape[:2]
print(f"\nDetections found: {len(boxes)}")

annotated = enhanced.copy()
for box, logit, phrase in zip(boxes.tolist(), logits.tolist(), phrases):
    cx, cy, bw, bh = box
    x1 = int((cx - bw/2) * w)
    y1 = int((cy - bh/2) * h)
    x2 = int((cx + bw/2) * w)
    y2 = int((cy + bh/2) * h)

    # Skip boxes covering more than 40% of image
    box_area = (x2 - x1) * (y2 - y1)
    img_area = w * h
    if box_area / img_area > 0.40:
        print(f"  SKIPPED (too large): {phrase} | conf: {logit:.3f}")
        continue

    print(f"  {phrase} | confidence: {logit:.3f} | box: [{x1},{y1},{x2},{y2}]")
    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,165,255), 2)
    cv2.putText(annotated, f"{phrase} {logit:.2f}", (x1, y1-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

cv2.imwrite("C:/Users/Cctech/TDS/outputs/test_detection.jpg", annotated)
print("\nSaved annotated image to outputs/test_detection.jpg")
