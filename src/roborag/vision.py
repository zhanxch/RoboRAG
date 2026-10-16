"""Frozen ImageNet ResNet-18 visual baseline; no task-specific training."""

import base64
from io import BytesIO

import numpy as np
from PIL import Image


class VisualEncoder:
    name = "torchvision-resnet18-IMAGENET1K_V1-global-3view"

    def __init__(self):
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        torch.set_num_threads(4)
        self.torch = torch
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
        self.model.fc = torch.nn.Identity()
        self.model.requires_grad_(False)

    def encode(self, image_urls, cameras):
        tensors = []
        for camera in cameras:
            raw = base64.b64decode(image_urls[camera].split(",", 1)[1], validate=True)
            with Image.open(BytesIO(raw)) as image:
                tensors.append(self.transform(image.convert("RGB")))
        with self.torch.inference_mode():
            features = self.model(self.torch.stack(tensors)).numpy()
        return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)


def distance(visual_a, state_a, visual_b, state_b):
    visual_cost = float(np.mean(1 - np.sum(np.asarray(visual_a) * np.asarray(visual_b), axis=1)))
    state_cost = 0.
    for arm in ("left", "right"):
        a = np.asarray(state_a[f"{arm}_ee_pose"], dtype=float)
        b = np.asarray(state_b[f"{arm}_ee_pose"], dtype=float)
        state_cost += min(float(np.linalg.norm(a[:3] - b[:3])) / .15, 3.)
        qa, qb = a[3:7], b[3:7]
        cosine = abs(float(np.dot(qa, qb))) / max(float(np.linalg.norm(qa) * np.linalg.norm(qb)), 1e-8)
        state_cost += float(2 * np.arccos(np.clip(cosine, 0, 1)) / np.pi)
        state_cost += .5 * abs(float(state_a[f"{arm}_ee_joint_state"][0]) - float(state_b[f"{arm}_ee_joint_state"][0]))
    return 3 * visual_cost + .5 * state_cost

