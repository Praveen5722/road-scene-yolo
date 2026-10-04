"""Project-wide constants."""
ROAD_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign", "van", "pedestrian", "cyclist", "tram"}
VULNERABLE = {"person", "bicycle", "motorcycle", "pedestrian", "cyclist"}
VEHICLES = {"car", "bus", "truck", "van", "tram"}
DEFAULT_WEIGHTS = "yolov8n.pt"
DEFAULT_CONF = 0.35
DEFAULT_IMGSZ = 640
NEAR_BOTTOM = 0.75
NEAR_HEIGHT = 0.35
