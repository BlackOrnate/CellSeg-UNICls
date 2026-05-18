from __future__ import annotations

from torch import nn


class MLPClassifier(nn.Module):
    """Lightweight classifier used after frozen UNI2 embedding extraction."""

    def __init__(self, in_dim: int = 1536, num_classes: int = 7, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)
