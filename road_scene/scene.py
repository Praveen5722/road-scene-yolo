"""Rule-based scene understanding on top of raw detections."""
from typing import List
import cv2
from .config import VULNERABLE, VEHICLES, NEAR_BOTTOM, NEAR_HEIGHT
from .schema import Detection, SceneObject, Alert, SceneReport
_ORDER = {"danger": 0, "warning": 1, "info": 2}

def zone_of(box, width) -> str:
    cx = (box[0] + box[2]) / 2 / width
    return "left" if cx < 1 / 3 else ("ahead" if cx < 2 / 3 else "right")

def is_near(box, height) -> bool:
    return (box[3] / height > NEAR_BOTTOM) or ((box[3] - box[1]) / height > NEAR_HEIGHT)

def analyze_scene(detections: List[Detection], shape) -> SceneReport:
    H, W = shape[:2]
    rep = SceneReport()
    alerts = {}
    for d in detections:
        zone = zone_of(d.box, W)
        near = is_near(d.box, H)
        rep.objects.append(SceneObject(d.name, zone, "near" if near else "far", d.conf))
        rep.counts[d.name] = rep.counts.get(d.name, 0) + 1
        if d.name in VULNERABLE and zone == "ahead" and near:
            alerts[f"BRAKE: {d.name} close ahead"] = "danger"
        elif d.name in VULNERABLE and zone == "ahead":
            alerts[f"CAUTION: {d.name} ahead"] = "warning"
        elif d.name in VEHICLES and zone == "ahead" and near:
            alerts[f"KEEP DISTANCE: {d.name} close ahead"] = "warning"
        elif d.name == "stop sign":
            alerts["Stop sign detected"] = "info"
        elif d.name == "traffic light":
            alerts["Traffic light detected"] = "info"
    rep.alerts = sorted((Alert(level, message) for message, level in alerts.items()), key=lambda a: (_ORDER[a.level], a.message))
    rep.summary = ("Scene: " + ", ".join(f"{v} {k}" for k, v in rep.counts.items()) if rep.counts else "No road objects detected.")
    return rep

def draw_zones(img_bgr):
    out = img_bgr.copy()
    H, W = out.shape[:2]
    for x in (W // 3, 2 * W // 3):
        cv2.line(out, (x, 0), (x, H), (255, 255, 0), 2)
    y = int(NEAR_BOTTOM * H)
    cv2.line(out, (0, y), (W, y), (0, 165, 255), 2)
    for label, x in (("LEFT", 8), ("AHEAD", W // 3 + 8), ("RIGHT", 2 * W // 3 + 8)):
        cv2.putText(out, label, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(out, "near line", (8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    return out
