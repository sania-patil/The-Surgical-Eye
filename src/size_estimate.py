"""
Stone size estimation from segmentation mask
Uses ureteroscope working channel diameter as reference (3.6Fr = ~1.2mm)
"""

import numpy as np
import cv2

# Calibration: typical ureteroscope field of view
# Working channel ~3.6Fr = 1.2mm, appears ~40-60px in standard endoscope view
# We use a conservative estimate: 1mm ≈ 30 pixels at standard zoom
PIXELS_PER_MM = 30.0


def estimate_size(mask: np.ndarray, pixels_per_mm: float = PIXELS_PER_MM) -> dict:
    """
    Estimate stone size from segmentation mask.
    Returns width, height, area in mm.
    """
    if mask is None or not np.any(mask):
        return {"width_mm": 0, "height_mm": 0, "area_mm2": 0, "diameter_mm": 0}

    # Get bounding box of mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    height_px = rmax - rmin
    width_px  = cmax - cmin
    area_px   = int(np.sum(mask))

    width_mm  = round(width_px  / pixels_per_mm, 1)
    height_mm = round(height_px / pixels_per_mm, 1)
    area_mm2  = round(area_px   / (pixels_per_mm ** 2), 1)

    # Equivalent diameter (circle with same area)
    diameter_mm = round(2 * np.sqrt(area_mm2 / np.pi), 1)

    return {
        "width_mm"   : width_mm,
        "height_mm"  : height_mm,
        "area_mm2"   : area_mm2,
        "diameter_mm": diameter_mm,
        "width_px"   : int(width_px),
        "height_px"  : int(height_px),
        "area_px"    : area_px
    }


def get_size_label(diameter_mm: float) -> str:
    """Clinical size classification."""
    if diameter_mm <= 0:
        return "Unknown"
    elif diameter_mm < 4:
        return "Small (<4mm)"
    elif diameter_mm < 10:
        return "Medium (4-10mm)"
    else:
        return "Large (>10mm)"
