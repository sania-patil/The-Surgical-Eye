import sys
sys.path.insert(0, "C:/Users/Cctech/TDS/src")

import cv2
from enhance import enhance_frame, get_visibility_score
from pathlib import Path

# Pick one training image
img_path = "C:/Users/Cctech/TDS/dataset/train/002-cropped_mp4-0000_jpg.rf.7S4lA0Dwr1u8u50EZ919.jpg"

frame = cv2.imread(img_path)
if frame is None:
    print("Image not found!")
else:
    print(f"Image loaded: {frame.shape}")
    enhanced = enhance_frame(frame)
    vis = get_visibility_score(enhanced)
    print(f"Enhanced shape: {enhanced.shape}")
    print(f"Visibility: {vis}")

    cv2.imwrite("C:/Users/Cctech/TDS/outputs/test_raw.jpg", frame)
    cv2.imwrite("C:/Users/Cctech/TDS/outputs/test_enhanced.jpg", enhanced)
    print("Saved raw and enhanced to outputs/")
