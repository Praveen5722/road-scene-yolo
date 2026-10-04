"""Detect + analyse a single image."""
import _path  # noqa: F401
import argparse
import cv2
from road_scene.detector import RoadDetector
from road_scene.scene import analyze_scene, draw_zones

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="annotated.jpg")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--zones", action="store_true")
    a = ap.parse_args()
    det = RoadDetector(a.weights, a.conf, a.imgsz)
    res = det.detect(a.input)
    rep = analyze_scene(res.detections, res.shape)
    cv2.imwrite(a.output, draw_zones(res.annotated) if a.zones else res.annotated)
    print(f"{rep.summary} ({res.inference_ms:.0f} ms)")
    for o in rep.objects:
        print(f"  {o.name:14s} {o.zone:6s} {o.distance:5s} {o.conf:.2f}")
    for al in rep.alerts:
        print(f"  [{al.level.upper()}] {al.message}")
    print("saved ->", a.output)

if __name__ == "__main__":
    main()
