"""
RIRS Kidney Stone Surgery - AI Guidance Dashboard
Streamlit UI for visualizing detection results
"""

import streamlit as st
import cv2
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Paths
OUTPUT_FRAMES  = Path("D:/TDS/outputs/annotated_frames")
OUTPUT_VIDEOS  = Path("D:/TDS/outputs/annotated_videos")
RESULTS_JSON   = Path("D:/TDS/outputs/results.json")
TEST_VIDEOS    = Path("D:/TDS/dataset/test_videos")

st.set_page_config(
    page_title="RIRS AI Assistant",
    page_icon="🔬",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.safe    { color: #00ff88; font-weight: bold; font-size: 1.2em; }
.unsafe  { color: #ff4444; font-weight: bold; font-size: 1.2em; }
.uncertain { color: #aaaaaa; font-weight: bold; font-size: 1.2em; }
.metric-box { background: #1e1e2e; padding: 15px; border-radius: 10px; margin: 5px; }
</style>
""", unsafe_allow_html=True)


def load_results():
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            return json.load(f)
    return {}


def load_frame(path):
    img = cv2.imread(str(path))
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 RIRS Kidney Stone Surgery — AI Visual Assistant")
st.markdown("**Real-time stone detection, size estimation & laser alignment safety**")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/kidney.png", width=80)
    st.header("Controls")

    mode = st.radio("Mode", ["Live Inference", "Results Viewer", "Frame Explorer"])

    st.divider()
    st.markdown("**Model Info**")
    st.markdown("- Detector: Grounding DINO")
    st.markdown("- Segmentor: SAM ViT-B")
    st.markdown("- Device: CUDA (RTX 3050)")
    st.divider()
    st.markdown("**Dataset**")
    st.markdown("- Train: 3,500+ frames")
    st.markdown("- Test: 2 videos")

# ── Main Content ──────────────────────────────────────────────────────────────

if mode == "Live Inference":
    st.header("▶ Live Video Inference")

    video_files = list(TEST_VIDEOS.glob("*.mp4"))
    if not video_files:
        st.warning("No test videos found in D:/TDS/dataset/test_videos/")
    else:
        selected = st.selectbox("Select test video", [v.name for v in video_files])
        video_path = TEST_VIDEOS / selected

        col1, col2 = st.columns([2, 1])

        with col1:
            if st.button("▶ Run Pipeline", type="primary", use_container_width=True):
                with st.spinner("Running AI pipeline on video..."):
                    sys.path.insert(0, "D:/TDS/src")
                    from pipeline import process_video
                    results = process_video(str(video_path), frame_interval=15)
                    st.success(f"Done! Processed {len(results)} keyframes")
                    st.rerun()

        with col2:
            ann_video = OUTPUT_VIDEOS / f"{video_path.stem}_annotated.mp4"
            if ann_video.exists():
                st.success("Annotated video ready")
                st.video(str(ann_video))

elif mode == "Results Viewer":
    st.header("📊 Results Analysis")

    results = load_results()
    if not results:
        st.info("No results yet. Run the pipeline first from 'Live Inference' tab.")
        st.code("D:\\TDS\\venv\\Scripts\\python.exe D:\\TDS\\src\\pipeline.py")
    else:
        for video_name, frames in results.items():
            st.subheader(f"🎬 {video_name}")

            if not frames:
                continue

            # Summary metrics
            total_frames   = len(frames)
            stone_frames   = sum(1 for f in frames if f.get("stones"))
            safe_frames    = sum(1 for f in frames if f.get("alignment", {}).get("status") == "Safe")
            unsafe_frames  = sum(1 for f in frames if f.get("alignment", {}).get("status") == "Not Safe")
            uncertain_frames = sum(1 for f in frames if f.get("alignment", {}).get("status") == "Uncertain")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Frames Analyzed", total_frames)
            c2.metric("Frames with Stone", stone_frames)
            c3.metric("✅ Safe Frames", safe_frames)
            c4.metric("❌ Not Safe Frames", unsafe_frames)

            # Laser alignment pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=["Safe", "Not Safe", "Uncertain"],
                values=[safe_frames, unsafe_frames, uncertain_frames],
                marker_colors=["#00ff88", "#ff4444", "#aaaaaa"]
            )])
            fig_pie.update_layout(title="Laser Alignment Distribution", height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

            # Stone size over time
            sizes = [f.get("size", {}).get("diameter_mm", 0) for f in frames]
            frame_idxs = [f.get("frame_idx", i) for i, f in enumerate(frames)]

            if any(s > 0 for s in sizes):
                df_size = pd.DataFrame({"Frame": frame_idxs, "Diameter (mm)": sizes})
                fig_size = px.line(df_size, x="Frame", y="Diameter (mm)",
                                   title="Stone Size Estimation Over Time",
                                   color_discrete_sequence=["#00aaff"])
                st.plotly_chart(fig_size, use_container_width=True)

            # Visibility over time
            vis_scores = [f.get("visibility", {}).get("overall", 0) for f in frames]
            df_vis = pd.DataFrame({"Frame": frame_idxs, "Visibility Score": vis_scores})
            fig_vis = px.area(df_vis, x="Frame", y="Visibility Score",
                              title="Visibility Quality Over Time",
                              color_discrete_sequence=["#ffaa00"])
            fig_vis.add_hline(y=0.6, line_dash="dash", line_color="green", annotation_text="Good threshold")
            fig_vis.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="Poor threshold")
            st.plotly_chart(fig_vis, use_container_width=True)

            st.divider()

elif mode == "Frame Explorer":
    st.header("🖼 Frame Explorer")

    results = load_results()
    if not results:
        st.info("No results yet. Run the pipeline first.")
    else:
        video_name = st.selectbox("Select Video", list(results.keys()))
        frames     = results[video_name]

        if frames:
            frame_dir = OUTPUT_FRAMES / Path(video_name).stem

            frame_idx = st.slider("Frame", 0, len(frames) - 1, 0)
            frame_data = frames[frame_idx]

            # Load annotated frame
            ann_path = frame_dir / f"frame_{frame_data['frame_idx']:06d}.jpg"
            raw_path = frame_dir / f"frame_{frame_data['frame_idx']:06d}_raw.jpg"

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Raw Frame**")
                if raw_path.exists():
                    st.image(load_frame(raw_path), use_column_width=True)
                else:
                    st.info("Raw frame not available")

            with col2:
                st.markdown("**Annotated Frame**")
                if ann_path.exists():
                    st.image(load_frame(ann_path), use_column_width=True)
                else:
                    st.info("Annotated frame not available")

            # Frame details
            st.markdown("---")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**Stones Detected**")
                stones = frame_data.get("stones", [])
                if stones:
                    for i, s in enumerate(stones):
                        st.markdown(f"Stone {i+1}: conf={s['confidence']:.2f}")
                else:
                    st.markdown("None detected")

            with c2:
                st.markdown("**Size Estimation**")
                size = frame_data.get("size", {})
                if size.get("diameter_mm", 0) > 0:
                    st.markdown(f"Diameter: **{size['diameter_mm']} mm**")
                    st.markdown(f"Width: {size['width_mm']} mm")
                    st.markdown(f"Height: {size['height_mm']} mm")
                    st.markdown(f"Area: {size['area_mm2']} mm²")
                else:
                    st.markdown("N/A")

            with c3:
                st.markdown("**Laser Alignment**")
                align = frame_data.get("alignment", {})
                status = align.get("status", "Unknown")
                reason = align.get("reason", "")
                if status == "Safe":
                    st.markdown(f'<p class="safe">✅ {status}</p>', unsafe_allow_html=True)
                elif status == "Not Safe":
                    st.markdown(f'<p class="unsafe">❌ {status}</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="uncertain">⚠ {status}</p>', unsafe_allow_html=True)
                st.markdown(f"_{reason}_")

                vis = frame_data.get("visibility", {})
                st.markdown(f"Visibility: **{vis.get('label', 'N/A')}** ({vis.get('overall', 0):.2f})")
