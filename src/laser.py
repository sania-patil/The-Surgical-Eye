"""
Laser detection and alignment safety classification
Detects laser fiber tip as brightest pixel cluster
Classifies: Safe / Not Safe / Uncertain
"""

import cv2
import numpy as np


def detect_laser(frame: np.ndarray) -> dict:
    """
    Detect laser tip in frame using brightness thresholding.
    Returns laser tip position and confidence.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold for very bright pixels (laser appears as bright spot)
    _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
    bright_mask = cv2.dilate(bright_mask, kernel, iterations=1)

    # Find contours of bright regions
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"detected": False, "tip": None, "confidence": 0.0, "area": 0}

    # Filter contours by area (laser tip is small but not tiny)
    valid = [c for c in contours if 10 < cv2.contourArea(c) < 2000]

    if not valid:
        return {"detected": False, "tip": None, "confidence": 0.0, "area": 0}

    # Pick brightest/largest contour
    largest = max(valid, key=cv2.contourArea)
    M = cv2.moments(largest)

    if M["m00"] == 0:
        return {"detected": False, "tip": None, "confidence": 0.0, "area": 0}

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    area = cv2.contourArea(largest)

    # Confidence based on brightness at tip
    tip_brightness = float(gray[cy, cx]) / 255.0

    return {
        "detected"  : True,
        "tip"       : (cx, cy),
        "confidence": round(tip_brightness, 3),
        "area"      : int(area),
        "contour"   : largest
    }


def classify_laser_alignment(
    laser_result: dict,
    stone_boxes: list,
    visibility_score: float,
    frame_shape: tuple
) -> dict:
    """
    Classify laser alignment safety.
    Returns: Safe / Not Safe / Uncertain + reason
    """
    # Low visibility → Uncertain
    if visibility_score < 0.3:
        return {
            "status"    : "Uncertain",
            "reason"    : "Poor visibility - cannot assess alignment",
            "confidence": 0.0,
            "color"     : (128, 128, 128)
        }

    # No laser detected
    if not laser_result["detected"]:
        return {
            "status"    : "Uncertain",
            "reason"    : "Laser not visible in frame",
            "confidence": 0.0,
            "color"     : (128, 128, 128)
        }

    # No stone detected
    if not stone_boxes:
        return {
            "status"    : "Not Safe",
            "reason"    : "No stone detected - laser target unclear",
            "confidence": laser_result["confidence"],
            "color"     : (0, 0, 255)
        }

    tip = laser_result["tip"]
    tx, ty = tip

    # Check if laser tip is inside any stone bounding box
    for box in stone_boxes:
        x1, y1, x2, y2 = box["box_xyxy"]

        # Expand box slightly for tolerance
        margin = 20
        x1e, y1e = x1 - margin, y1 - margin
        x2e, y2e = x2 + margin, y2 + margin

        if x1e <= tx <= x2e and y1e <= ty <= y2e:
            # Laser inside stone box
            conf = min(laser_result["confidence"] * box["confidence"], 1.0)
            return {
                "status"    : "Safe",
                "reason"    : "Laser aligned with stone",
                "confidence": round(conf, 3),
                "color"     : (0, 255, 0)
            }

    # Laser detected but not on stone
    return {
        "status"    : "Not Safe",
        "reason"    : "Laser aimed away from stone",
        "confidence": laser_result["confidence"],
        "color"     : (0, 0, 255)
    }


def draw_laser(frame: np.ndarray, laser_result: dict, alignment: dict) -> np.ndarray:
    """Draw laser tip and alignment status on frame."""
    annotated = frame.copy()

    if laser_result["detected"]:
        tx, ty = laser_result["tip"]
        color  = alignment["color"]

        # Draw crosshair at laser tip
        cv2.drawMarker(annotated, (tx, ty), color,
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.circle(annotated, (tx, ty), 10, color, 2)

    return annotated
