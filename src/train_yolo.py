"""
Fine-tune YOLOv8n on auto-labeled kidney stone images
Run AFTER auto_label.py completes
"""

from ultralytics import YOLO
from pathlib import Path
import shutil

DATASET_YAML = "C:/Users/Cctech/TDS/dataset.yaml"
MODEL_OUTPUT  = "D:/TDS/outputs/yolo_model"

def train():
    print("Loading YOLOv8n base model...")
    model = YOLO("yolov8n.pt")  # downloads ~6MB nano model

    print("Starting fine-tuning...")
    results = model.train(
        data=DATASET_YAML,
        epochs=30,
        imgsz=416,
        batch=4,
        device=0,
        project=MODEL_OUTPUT,
        name="kidney_stone",
        patience=10,
        save=True,
        plots=True,
        verbose=True,
        workers=0
    )

    best_model = Path(MODEL_OUTPUT) / "kidney_stone" / "weights" / "best.pt"
    print(f"\nTraining complete!")
    print(f"Best model saved at: {best_model}")
    return str(best_model)

if __name__ == "__main__":
    train()
