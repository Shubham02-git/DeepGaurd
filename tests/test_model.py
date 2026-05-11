import importlib
import sys
import types

import numpy as np
import torch


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
    def __init__(self, logits):
        self.logits = logits
        self.last_output_shape = None

    def __call__(self, data_dict, inference=False):
        self.last_output_shape = self.logits.shape
        return {"cls": self.logits}


def test_model_inference_uses_mocked_logits_shape(monkeypatch):
    api = _import_api()
    model = DummyModel(torch.tensor([[0.1, 0.9]], dtype=torch.float32))
    monkeypatch.setitem(api.LOADED_MODELS, "xception", model)

    result = api.predict_tensor(torch.zeros((1, 3, 256, 256)), "xception")

    assert model.last_output_shape == (1, 2)
    assert result["label"] == "fake"
    assert "confidence" in result


def test_predict_tensor_selects_real_label_from_probabilities(monkeypatch):
    api = _import_api()
    monkeypatch.setitem(
        api.LOADED_MODELS,
        "xception",
        DummyModel(torch.tensor([[4.0, 1.0]], dtype=torch.float32)),
    )

    result = api.predict_tensor(torch.zeros((1, 3, 256, 256)), "xception")

    assert result["label"] == "real"
    assert result["prob_real"] > result["prob_fake"]


def test_predict_tensor_selects_fake_label_from_probabilities(monkeypatch):
    api = _import_api()
    monkeypatch.setitem(
        api.LOADED_MODELS,
        "xception",
        DummyModel(torch.tensor([[1.0, 4.0]], dtype=torch.float32)),
    )

    result = api.predict_tensor(torch.zeros((1, 3, 256, 256)), "xception")

    assert result["label"] == "fake"
    assert result["prob_fake"] > result["prob_real"]
