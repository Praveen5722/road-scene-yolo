"""Fine-tune YOLOv8 on a driving dataset.

Example: python scripts/train.py --data data.yaml --epochs 50 --imgsz 640
"""
import _path  # noqa: F401
import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", default="road_ft")
    a = ap.parse_args()
    m = YOLO(a.weights)
    m.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, fraction=a.fraction, device=a.device, workers=2, plots=True, project="runs", name=a.name, exist_ok=True)
    print("Training complete. Best weights are under runs/detect/road_ft/weights/best.pt")

if __name__ == "__main__":
    main()
