import abc
import torch
from typing import Union

class AbstractBackbone(abc.ABC):
    def __init__(self, config, load_param: Union[bool, str] = False):
        pass

    @abc.abstractmethod
    def features(self, data_dict: dict) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def init_weights(self, pretrained_path: Union[bool, str]):
        pass
