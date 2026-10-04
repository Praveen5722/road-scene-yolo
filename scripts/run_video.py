"""Annotate a video and report counts."""
import _path  # noqa: F401
import argparse
from road_scene.detector import RoadDetector
from road_scene.video import process_video, count_unique

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="annotated.mp4")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--unique", action="store_true")
    a = ap.parse_args()
    det = RoadDetector(a.weights, a.conf, a.imgsz)
    out = process_video(det, a.input, a.output, a.stride, a.max_frames)
    print(f"processed {len(out['counts'])} frames -> {out['out_path']}")
    if a.unique:
        print("unique objects:", count_unique(det, a.input, a.max_frames))

if __name__ == "__main__":
    main()
