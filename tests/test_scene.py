"""Offline tests for scene logic."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from road_scene.schema import Detection
from road_scene.scene import analyze_scene, zone_of, is_near, draw_zones
SHAPE = (1000, 900)

def test_zones():
    assert zone_of((0, 0, 100, 100), 900) == "left"
    assert zone_of((400, 0, 500, 100), 900) == "ahead"
    assert zone_of((800, 0, 899, 100), 900) == "right"

def test_near_by_bottom_and_height():
    assert is_near((0, 0, 10, 800), 1000)
    assert is_near((0, 100, 10, 500), 1000)
    assert not is_near((0, 100, 10, 200), 1000)

def test_person_ahead_near_triggers_brake():
    rep = analyze_scene([Detection("person", 0.9, (400, 300, 500, 900))], SHAPE)
    assert rep.alerts[0].level == "danger" and "BRAKE" in rep.alerts[0].message

def test_person_ahead_far_is_caution():
    rep = analyze_scene([Detection("person", 0.9, (400, 100, 430, 160))], SHAPE)
    assert rep.alerts[0].level == "warning" and "CAUTION" in rep.alerts[0].message

def test_vehicle_ahead_near_keep_distance():
    ahead = analyze_scene([Detection("car", 0.9, (350, 400, 550, 900))], SHAPE)
    side = analyze_scene([Detection("car", 0.9, (10, 400, 150, 900))], SHAPE)
    assert "KEEP DISTANCE" in ahead.alerts[0].message
    assert side.alerts == []

def test_info_alerts_and_sorting():
    dets = [Detection("traffic light", 0.8, (10, 10, 30, 60)), Detection("person", 0.9, (400, 300, 500, 900))]
    rep = analyze_scene(dets, SHAPE)
    assert [a.level for a in rep.alerts] == ["danger", "info"]

def test_counts_and_summary():
    dets = [Detection("car", 0.9, (0, 0, 10, 10)), Detection("car", 0.8, (20, 0, 30, 10)), Detection("person", 0.7, (50, 0, 60, 10))]
    rep = analyze_scene(dets, SHAPE)
    assert rep.counts == {"car": 2, "person": 1}
    assert rep.summary == "Scene: 2 car, 1 person"

def test_empty_scene():
    rep = analyze_scene([], SHAPE)
    assert rep.summary == "No road objects detected." and rep.alerts == []

def test_draw_zones_keeps_shape():
    img = np.zeros((300, 400, 3), np.uint8)
    out = draw_zones(img)
    assert out.shape == img.shape and out.sum() > 0
