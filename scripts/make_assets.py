"""Regenerate architecture and sample output assets.

Requires YOLO weights and the sample image. Run from the repository root:
    python scripts/make_assets.py
"""
import _path  # noqa: F401
import json
import time
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from road_scene.detector import RoadDetector
from road_scene.scene import analyze_scene, draw_zones

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "assets"
OUT = A / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

def architecture():
    fig, ax = plt.subplots(figsize=(15.6, 5.2))
    ax.set_xlim(0, 15.6); ax.set_ylim(0, 5.2); ax.axis("off")
    def box(x, y, w, h, text):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15", fc="#e6f2d9", ec="#333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#333"))
    y, h, w = 2.7, 1.5, 2.15
    xs = [0.2, 2.75, 5.3, 7.85, 10.4, 12.95]
    labels = ["Input\nImage or Video", "Pre-processing\nResize", "YOLOv8 CNN\nboxes + classes", "Post-processing\nroad-class filter", "Scene Analysis\nzone / distance / hazards", "Output\nannotations + alerts"]
    for x, t in zip(xs, labels): box(x, y, w, h, t)
    for i in range(5): arrow(xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)
    ax.text(7.8, 4.85, "Road Scene Understanding: system architecture", ha="center", fontsize=14, fontweight="bold")
    fig.savefig(A / "architecture.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def sample_outputs():
    det = RoadDetector("yolov8n.pt")
    img = cv2.imread(str(A / "sample_street.jpg"))
    if img is None:
        raise FileNotFoundError("assets/sample_street.jpg is missing")
    res = det.detect(img)
    cv2.imwrite(str(OUT / "sample_detection.jpg"), res.annotated)
    cv2.imwrite(str(OUT / "sample_zones.jpg"), draw_zones(res.annotated))
    return det, img, analyze_scene(res.detections, res.shape)

def benchmark(det, img):
    rows = []
    for sz in (320, 416, 640):
        det.imgsz = sz; det.detect(img)
        timings = []
        for _ in range(5):
            t0 = time.perf_counter(); r = det.detect(img); timings.append((time.perf_counter() - t0) * 1000)
        rows.append({"imgsz": sz, "ms": round(sum(timings) / len(timings), 1), "objects": len(r.detections)})
    det.imgsz = 640
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar([str(r["imgsz"]) for r in rows], [r["ms"] for r in rows])
    ax.set_xlabel("inference size (px)"); ax.set_ylabel("ms per image (CPU)"); ax.set_title("YOLOv8n speed vs input size")
    fig.tight_layout(); fig.savefig(OUT / "benchmark.png", dpi=150); plt.close(fig)
    return rows

def coco8_val():
    from ultralytics import YOLO
    v = YOLO("yolov8n.pt").val(data="coco8.yaml", imgsz=640, device="cpu", plots=False, verbose=False)
    return {"precision": round(float(v.box.mp), 3), "recall": round(float(v.box.mr), 3), "map50": round(float(v.box.map50), 3), "map50_95": round(float(v.box.map), 3)}

if __name__ == "__main__":
    architecture(); det, img, rep = sample_outputs(); bench = benchmark(det, img); val = coco8_val()
    results = {"sample_summary": rep.summary, "alerts": [a.message for a in rep.alerts], "benchmark_cpu": bench, "coco8_val": val}
    (ROOT / "docs" / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
