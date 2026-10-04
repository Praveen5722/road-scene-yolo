"""Thin wrapper around Ultralytics YOLO that keeps only road-related classes."""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List
import cv2
import numpy as np
from ultralytics import YOLO
from .config import ROAD_CLASSES, DEFAULT_WEIGHTS, DEFAULT_CONF, DEFAULT_IMGSZ
from .schema import Detection

@dataclass
class DetectionResult:
    detections: List[Detection]
    annotated: np.ndarray
    shape: tuple
    inference_ms: float

class RoadDetector:
    def __init__(self, weights=DEFAULT_WEIGHTS, conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ):
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        ids = [i for i, n in self.model.names.items() if n in ROAD_CLASSES]
        self.class_ids = ids or None

    def detect(self, image) -> DetectionResult:
        if isinstance(image, (str, Path)):
            path = str(image)
            image = cv2.imread(path)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {path}")
        t0 = time.perf_counter()
        r = self.model(image, conf=self.conf, imgsz=self.imgsz, classes=self.class_ids, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        dets = [Detection(r.names[int(b.cls)], float(b.conf), tuple(float(v) for v in b.xyxy[0].tolist())) for b in r.boxes]
        return DetectionResult(dets, r.plot(), image.shape[:2], ms)
