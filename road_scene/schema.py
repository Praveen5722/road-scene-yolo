"""Plain data containers shared across the package."""
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Detection:
    name: str
    conf: float
    box: Tuple[float, float, float, float]

@dataclass
class SceneObject:
    name: str
    zone: str
    distance: str
    conf: float

@dataclass
class Alert:
    level: str
    message: str

@dataclass
class SceneReport:
    objects: List[SceneObject] = field(default_factory=list)
    alerts: List[Alert] = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    summary: str = ""
