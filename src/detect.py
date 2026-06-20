"""
Stone detection using Grounding DINO (zero-shot)
Falls back to YOLOv8 if fine-tuned model is available
"""

import torch
import numpy as np
import cv2
from pathlib import Path

# Grounding DINO
from groundingdino.util.inference import load_model, load_image, predict
from PIL import Image

# Config
CONFIG_PATH  = "D:/TDS/groundingdino_config.py"
WEIGHTS_PATH = "D:/TDS/groundingdino_swint_ogc.pth"
YOLO_MODEL   = "D:/TDS/outputs/yolo_model/kidney_stone/weights/best.pt"
TEXT_PROMPT  = "kidney stone . calculus . renal stone"
BOX_THRESHOLD  = 0.30
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_dino_model = None
_yolo_model = None


def load_detector():
    """Load Grounding DINO model (singleton)."""
    global _dino_model
    if _dino_model is None:
        print("Loading Grounding DINO...")
        _dino_model = load_model(CONFIG_PATH, WEIGHTS_PATH, device=DEVICE)
        print("Grounding DINO loaded.")
    return _dino_model


def load_yolo():
    """Load fine-tuned YOLO model if available."""
    global _yolo_model
    if _yolo_model is None and Path(YOLO_MODEL).exists():
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_MODEL)
        print("Fine-tuned YOLO loaded.")
    return _yolo_model


def detect_stones_dino(frame: np.ndarray) -> list:
    """
    Detect kidney stones using Grounding DINO.
    Returns list of dicts: {box_xyxy, confidence, label}
    """
    model = load_detector()
    h, w  = frame.shape[:2]

    # Convert BGR to RGB PIL image for DINO
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil   = Image.fromarray(rgb)

    # Save temp and reload via DINO loader
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        pil.save(tmp_path)

    try:
        image_source, image = load_image(tmp_path)
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE
        )
    finally:
        os.unlink(tmp_path)

    detections = []
    for box, logit, phrase in zip(boxes.tolist(), logits.tolist(), phrases):
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        detections.append({
            "box_xyxy": [x1, y1, x2, y2],
            "confidence": round(logit, 3),
            "label": phrase
        })

    return detections


def detect_stones_yolo(frame: np.ndarray) -> list:
    """Detect using fine-tuned YOLO if available."""
    model = load_yolo()
    if model is None:
        return []

    results = model(frame, conf=0.3, device=DEVICE, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = round(float(box.conf[0]), 3)
            detections.append({
                "box_xyxy": [x1, y1, x2, y2],
                "confidence": conf,
                "label": "kidney_stone"
            })
    return detections


def detect_stones(frame: np.ndarray) -> list:
    """
    Main detection function.
    Uses YOLO if fine-tuned model exists, otherwise Grounding DINO.
    """
    yolo = load_yolo()
    if yolo is not None:
        return detect_stones_yolo(frame)
    return detect_stones_dino(frame)
