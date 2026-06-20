"""
Stone segmentation using SAM (Segment Anything Model)
Takes bounding box from detector and produces precise mask
"""

import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor

SAM_CHECKPOINT = "D:/TDS/sam_vit_b.pth"
SAM_MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_sam_predictor = None


def load_sam():
    """Load SAM predictor (singleton)."""
    global _sam_predictor
    if _sam_predictor is None:
        print("Loading SAM...")
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam.to(device=DEVICE)
        _sam_predictor = SamPredictor(sam)
        print("SAM loaded.")
    return _sam_predictor


def segment_stone(frame: np.ndarray, box_xyxy: list) -> dict:
    """
    Segment stone given bounding box.
    Returns dict with mask, area, and contour.
    """
    predictor = load_sam()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    box = np.array(box_xyxy)
    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=True
    )

    # Pick best mask
    best_idx  = np.argmax(scores)
    best_mask = masks[best_idx]
    best_score = scores[best_idx]

    # Get contours
    mask_uint8 = best_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area_pixels = int(np.sum(best_mask))

    return {
        "mask": best_mask,
        "score": round(float(best_score), 3),
        "area_pixels": area_pixels,
        "contours": contours
    }


def draw_segmentation(frame: np.ndarray, seg_result: dict, color=(0, 255, 100)) -> np.ndarray:
    """Overlay segmentation mask on frame."""
    annotated = frame.copy()
    mask = seg_result["mask"]

    # Color overlay
    overlay = annotated.copy()
    overlay[mask] = color
    annotated = cv2.addWeighted(annotated, 0.6, overlay, 0.4, 0)

    # Draw contour
    if seg_result["contours"]:
        cv2.drawContours(annotated, seg_result["contours"], -1, color, 2)

    return annotated
