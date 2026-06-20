# RIRS Kidney Stone Surgery - AI Visual Assistant

## Overview
AI-based visual assistant for Retrograde Intrarenal Surgery (RIRS) that detects kidney stones, estimates size, and classifies laser alignment safety in real-time endoscopic video.

## Pipeline
```
Input Video → Frame Extraction → CLAHE Enhancement → Denoising → 
Grounding DINO Detection → SAM Segmentation → Size Estimation → 
Laser Detection → Safety Classification → Annotated Video + Dashboard
```

## Features
- Stone detection and localization using Grounding DINO (zero-shot)
- Stone segmentation using SAM ViT-B
- Size estimation in mm
- Laser alignment classification: Safe / Not Safe / Uncertain
- Streamlit dashboard with charts and frame explorer

## Tech Stack
- Grounding DINO (zero-shot object detection)
- SAM ViT-B (Segment Anything Model)
- YOLOv8n (fine-tuned on auto-labeled data)
- OpenCV (image enhancement, laser detection)
- Streamlit + Plotly (dashboard)
- PyTorch + CUDA

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Run Pipeline
```bash
python src/pipeline.py
```

## Run Dashboard
```bash
streamlit run dashboard/app.py
```

## Results
- Annotated videos: `outputs/annotated_videos/`
- Annotated frames: `outputs/annotated_frames/`
- Results JSON: `outputs/results.json`
