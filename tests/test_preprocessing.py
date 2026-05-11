import importlib
import sys
import types

import numpy as np
import pytest
import torch
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


def test_preprocess_image_returns_expected_shape():
    api = _import_api()
    image = Image.new("RGB", (64, 48), color="white")

    tensor = api.preprocess_image(image)

    assert tensor.shape == (1, 3, 256, 256)
    assert tensor.dtype == torch.float32


def test_preprocess_image_rejects_invalid_input():
    api = _import_api()

    with pytest.raises(AttributeError):
        api.preprocess_image("not-an-image")


def test_extract_video_frames_returns_exactly_16_frames(monkeypatch, tmp_path):
    api = _import_api()

    class FakeVideoCapture:
        def __init__(self, path):
            self.current = 0

        def get(self, prop):
            return 32

        def set(self, prop, value):
            self.current = int(value)

        def read(self):
            frame = np.full((12, 12, 3), self.current, dtype=np.uint8)
            return True, frame

        def release(self):
            pass

    monkeypatch.setattr(api.cv2, "VideoCapture", FakeVideoCapture)
    monkeypatch.setattr(api.cv2, "cvtColor", lambda frame, code: frame)
    monkeypatch.setattr(api, "ROOT", str(tmp_path))

    frames = api.extract_video_frames(b"fake-video-bytes")

    assert len(frames) == 16
    assert all(isinstance(frame, Image.Image) for frame in frames)


def test_video_frame_tensor_shape_correctness(monkeypatch, tmp_path):
    api = _import_api()

    class FakeVideoCapture:
        def __init__(self, path):
            pass

        def get(self, prop):
            return 16

        def set(self, prop, value):
            pass

        def read(self):
            return True, np.zeros((16, 16, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(api.cv2, "VideoCapture", FakeVideoCapture)
    monkeypatch.setattr(api.cv2, "cvtColor", lambda frame, code: frame)
    monkeypatch.setattr(api, "ROOT", str(tmp_path))

    frames = api.extract_video_frames(b"fake-video-bytes")
    tensor = torch.cat([api.preprocess_image(frame) for frame in frames], dim=0)

    assert tensor.shape == (16, 3, 256, 256)


def test_extract_video_frames_rejects_invalid_video(monkeypatch, tmp_path):
    api = _import_api()
    monkeypatch.setattr(api, "ROOT", str(tmp_path))

    with pytest.raises(ValueError, match="Empty or unreadable video file"):
        api.extract_video_frames(b"not-a-real-video")
