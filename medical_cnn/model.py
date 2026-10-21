"""Regularized CNN for educational medical-image classification."""
import torch
from torch import nn

class RegularizedCNN(nn.Module):
    """Compact CNN with batch normalization and dropout."""
    def __init__(self, num_classes: int, dropout: float = 0.4) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout2d(p=dropout / 2),
            )
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, num_classes))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
