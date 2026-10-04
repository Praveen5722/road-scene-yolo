"""Validate a model: precision, recall, mAP."""
import _path  # noqa: F401
import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--data", default="coco8.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    m = YOLO(a.weights)
    v = m.val(data=a.data, imgsz=a.imgsz, device=a.device, plots=False, verbose=False)
    print(f"Precision {v.box.mp:.3f} | Recall {v.box.mr:.3f} | mAP50 {v.box.map50:.3f} | mAP50-95 {v.box.map:.3f}")
    for c, ap_ in zip(v.box.ap_class_index, v.box.maps[v.box.ap_class_index]):
        print(f"  {v.names[int(c)]:16s} mAP50-95 {ap_:.3f}")

if __name__ == "__main__":
    main()
