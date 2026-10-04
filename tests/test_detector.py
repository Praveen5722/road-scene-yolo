"""Integration test for YOLO detection; skips when weights cannot be loaded."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
SAMPLE = Path(__file__).resolve().parents[1] / "assets" / "sample_street.jpg"

def test_sample_image_has_bus_and_people():
    try:
        from road_scene.detector import RoadDetector
        det = RoadDetector("yolov8n.pt")
    except Exception as exc:
        pytest.skip(f"model unavailable: {exc}")
    res = det.detect(SAMPLE)
    names = {d.name for d in res.detections}
    assert "bus" in names and "person" in names
    assert res.annotated.shape[:2] == res.shape
