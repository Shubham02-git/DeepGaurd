import importlib
import io
import sys
import types

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image


def _import_api():
    if "torchvision.transforms" not in sys.modules:
        transforms = types.ModuleType("torchvision.transforms")

        class Compose:
            def __init__(self, steps):
                self.steps = steps

            def __call__(self, value):
                for step in self.steps:
                    value = step(value)
                return value

        class Resize:
            def __init__(self, size):
                self.size = size

            def __call__(self, image):
                return image.resize(self.size)

        class ToTensor:
            def __call__(self, image):
                array = np.asarray(image, dtype=np.float32) / 255.0
                return torch.from_numpy(array).permute(2, 0, 1)

        class Normalize:
            def __init__(self, mean, std):
                self.mean = torch.tensor(mean).view(3, 1, 1)
                self.std = torch.tensor(std).view(3, 1, 1)

            def __call__(self, tensor):
                return (tensor - self.mean) / self.std

        transforms.Compose = Compose
        transforms.Resize = Resize
        transforms.ToTensor = ToTensor
        transforms.Normalize = Normalize

        torchvision = types.ModuleType("torchvision")
        torchvision.transforms = transforms
        sys.modules["torchvision"] = torchvision
        sys.modules["torchvision.transforms"] = transforms

    if "detectors" not in sys.modules:
        detectors = types.ModuleType("detectors")

        class DummyDetector(torch.nn.Module):
            def __init__(self, config):
                super().__init__()

            def forward(self, data_dict, inference=False):
                return {"cls": torch.zeros((data_dict["image"].size(0), 2))}

        detectors.DETECTOR = {
            "xception": DummyDetector,
            "ucf": DummyDetector,
            "efficientnetb4": DummyDetector,
        }
        sys.modules["detectors"] = detectors

    return importlib.import_module("inference_api")


class DummyModel:
    def __call__(self, data_dict, inference=False):
        return {"cls": torch.tensor([[0.1, 0.9]], dtype=torch.float32)}


def _png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color="white").save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def test_predict_successful_image_upload(monkeypatch):
    api = _import_api()
    monkeypatch.setitem(api.LOADED_MODELS, "xception", DummyModel())
    monkeypatch.setattr(api, "is_whitelisted", lambda filename: False)
    client = TestClient(api.app)

    response = client.post(
        "/predict?model=xception",
        files={"file": ("sample.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "label" in payload
    assert "confidence" in payload
    assert payload["label"] in {"real", "fake"}
    assert isinstance(payload["confidence"], float)


def test_predict_rejects_invalid_file_type(monkeypatch):
    api = _import_api()
    monkeypatch.setitem(api.LOADED_MODELS, "xception", DummyModel())
    monkeypatch.setattr(api, "is_whitelisted", lambda filename: False)
    client = TestClient(api.app)

    response = client.post(
        "/predict?model=xception",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )

    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]
