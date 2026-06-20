"""
Frame extraction and image enhancement
CLAHE + denoising for endoscopic video frames
"""

import cv2
import numpy as np
from pathlib import Path


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE + denoising to a single frame."""
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Denoise
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)

    return enhanced


def extract_frames(video_path: str, output_dir: str, frame_interval: int = 5):
    """
    Extract frames from video at given interval.
    Returns list of (frame_idx, raw_frame, enhanced_frame) tuples.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS)
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {Path(video_path).name}")
    print(f"  FPS: {fps:.1f} | Frames: {total} | Size: {width}x{height}")

    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            enhanced = enhance_frame(frame)

            # Save pre and post processing images
            raw_path = output_dir / f"frame_{frame_idx:05d}_raw.jpg"
            enh_path = output_dir / f"frame_{frame_idx:05d}_enhanced.jpg"
            cv2.imwrite(str(raw_path), frame)
            cv2.imwrite(str(enh_path), enhanced)

            frames.append((frame_idx, frame, enhanced))

        frame_idx += 1

    cap.release()
    print(f"  Extracted {len(frames)} frames (every {frame_interval} frames)")
    return frames, fps, (width, height)


def get_visibility_score(frame: np.ndarray) -> dict:
    """
    Compute visibility score based on blur and brightness.
    Returns score dict with 'blur', 'brightness', 'overall', 'label'
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur score (Laplacian variance - higher = sharper)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Brightness score
    brightness = np.mean(gray)

    # Normalize
    blur_norm       = min(blur_score / 500.0, 1.0)
    brightness_norm = 1.0 - abs(brightness - 128) / 128.0

    overall = (blur_norm * 0.6 + brightness_norm * 0.4)

    if overall > 0.6:
        label = "Good"
    elif overall > 0.3:
        label = "Poor"
    else:
        label = "Very Poor"

    return {
        "blur": round(blur_score, 2),
        "brightness": round(float(brightness), 2),
        "overall": round(overall, 3),
        "label": label
    }
