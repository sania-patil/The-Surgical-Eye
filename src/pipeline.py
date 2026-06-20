"""
Main pipeline orchestrator
Processes test videos end-to-end
"""

import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from enhance import enhance_frame, get_visibility_score
from detect import detect_stones
from segment import segment_stone, draw_segmentation
from size_estimate import estimate_size, get_size_label
from laser import detect_laser, classify_laser_alignment, draw_laser

# Paths
TEST_VIDEO_DIR   = Path("D:/TDS/dataset/test_videos")
OUTPUT_FRAMES    = Path("D:/TDS/outputs/annotated_frames")
OUTPUT_VIDEOS    = Path("D:/TDS/outputs/annotated_videos")
RESULTS_JSON     = Path("D:/TDS/outputs/results.json")

OUTPUT_FRAMES.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEOS.mkdir(parents=True, exist_ok=True)


def annotate_frame(frame: np.ndarray, frame_idx: int, video_name: str) -> dict:
    """Run full pipeline on a single frame."""
    result = {
        "frame_idx"  : frame_idx,
        "video"      : video_name,
        "stones"     : [],
        "laser"      : {},
        "alignment"  : {},
        "visibility" : {},
        "size"       : {}
    }

    # Step 1: Enhance
    enhanced = enhance_frame(frame)

    # Step 2: Visibility
    vis = get_visibility_score(enhanced)
    result["visibility"] = vis

    # Step 3: Detect stones
    stones = detect_stones(enhanced)
    result["stones"] = stones

    # Step 4: Segment + size (first stone only for speed)
    annotated = enhanced.copy()
    if stones:
        best_stone = max(stones, key=lambda x: x["confidence"])
        try:
            seg = segment_stone(enhanced, best_stone["box_xyxy"])
            size = estimate_size(seg["mask"])
            result["size"] = size
            result["seg_score"] = seg["score"]
            annotated = draw_segmentation(annotated, seg)
        except Exception as e:
            print(f"  Segmentation error frame {frame_idx}: {e}")
            size = {}

    # Step 5: Laser detection
    laser = detect_laser(enhanced)
    result["laser"] = {
        "detected"  : laser["detected"],
        "tip"       : laser["tip"],
        "confidence": laser["confidence"]
    }

    # Step 6: Alignment classification
    alignment = classify_laser_alignment(
        laser, stones, vis["overall"], frame.shape
    )
    result["alignment"] = alignment

    # Step 7: Draw annotations
    # Draw stone bounding boxes
    for stone in stones:
        x1, y1, x2, y2 = stone["box_xyxy"]
        conf = stone["confidence"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(annotated, f"Stone {conf:.2f}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    # Draw laser
    annotated = draw_laser(annotated, laser, alignment)

    # Draw HUD
    annotated = draw_hud(annotated, result)

    return result, annotated


def draw_hud(frame: np.ndarray, result: dict) -> np.ndarray:
    """Draw heads-up display with all info."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent panel
    cv2.rectangle(overlay, (0, 0), (320, 160), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    y = 22
    def put(text, color=(255, 255, 255)):
        nonlocal y
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
        y += 22

    stones = result.get("stones", [])
    vis    = result.get("visibility", {})
    size   = result.get("size", {})
    align  = result.get("alignment", {})

    put(f"Stones: {len(stones)}", (0, 200, 255))

    if size.get("diameter_mm", 0) > 0:
        label = get_size_label(size["diameter_mm"])
        put(f"Size: {size['diameter_mm']}mm ({label})", (0, 255, 200))
    else:
        put("Size: N/A")

    vis_label = vis.get("label", "N/A")
    vis_color = (0, 255, 0) if vis_label == "Good" else (0, 165, 255) if vis_label == "Poor" else (0, 0, 255)
    put(f"Visibility: {vis_label}", vis_color)

    status = align.get("status", "Uncertain")
    s_color = (0, 255, 0) if status == "Safe" else (0, 0, 255) if status == "Not Safe" else (128, 128, 128)
    put(f"Laser: {status}", s_color)

    reason = align.get("reason", "")
    if reason:
        put(reason[:40], (200, 200, 200))

    put(f"Frame: {result['frame_idx']}", (180, 180, 180))

    return frame


def process_video(video_path: str, frame_interval: int = 10) -> list:
    """Process a single video end-to-end."""
    video_path = Path(video_path)
    video_name = video_path.stem
    frame_dir  = OUTPUT_FRAMES / video_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nProcessing: {video_name}")
    print(f"  {total} frames @ {fps:.1f} FPS | {width}x{height}")

    out_path = OUTPUT_VIDEOS / f"{video_name}_annotated.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    results    = []
    frame_idx  = 0
    prev_annotated = None

    pbar = tqdm(total=total, desc=video_name)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            try:
                result, annotated = annotate_frame(frame, frame_idx, video_name)
                results.append(result)
                prev_annotated = annotated

                # Save annotated frame
                cv2.imwrite(str(frame_dir / f"frame_{frame_idx:06d}.jpg"), annotated)

                # Save raw frame for comparison
                cv2.imwrite(str(frame_dir / f"frame_{frame_idx:06d}_raw.jpg"), frame)

            except Exception as e:
                print(f"  Error frame {frame_idx}: {e}")
                annotated = frame

        # Write every frame to video (use last annotated for skipped frames)
        if prev_annotated is not None:
            out_frame = cv2.resize(prev_annotated, (width, height))
            writer.write(out_frame)
        else:
            writer.write(frame)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()

    print(f"  Annotated video: {out_path}")
    print(f"  Processed {len(results)} keyframes")
    return results


def run_pipeline():
    """Run full pipeline on all test videos."""
    video_files = list(TEST_VIDEO_DIR.glob("*.mp4"))
    if not video_files:
        print("No test videos found!")
        return

    all_results = {}

    for video_path in video_files:
        results = process_video(str(video_path), frame_interval=10)
        all_results[video_path.name] = results

    # Save results JSON
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, tuple): return list(obj)
        return obj

    import json
    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\nResults saved to: {RESULTS_JSON}")
    print("Pipeline complete!")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    run_pipeline()
