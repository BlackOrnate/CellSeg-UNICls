import torch.nn.functional as F
from torch import nn


class Classifier(nn.Module):
    def __init__(self, in_dim=1536, num_classes=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

        # self.net = nn.Sequential(
        #     nn.Linear(in_dim, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(256, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(32, num_classes),
        # )

    def forward(self, x):
        output = self.net(x)
        return output
