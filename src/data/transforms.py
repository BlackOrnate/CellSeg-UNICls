from __future__ import annotations

from torchvision import transforms


def build_image_transform(resize_to: int | None = 224):
    steps = []
    if resize_to is not None:
        steps.append(transforms.Resize((resize_to, resize_to)))
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)
