import abc
import torch
import torch.nn as nn
from typing import Union

class AbstractDetector(nn.Module, metaclass=abc.ABCMeta):
\
\
       
    def __init__(self, config=None, load_param: Union[bool, str] = False):
\
\
\
\
\
           
        super().__init__()

    @abc.abstractmethod
    def features(self, data_dict: dict) -> torch.tensor:
\
\
           
        pass

    @abc.abstractmethod
    def forward(self, data_dict: dict, inference=False) -> dict:
\
\
           
        pass

    @abc.abstractmethod
    def classifier(self, features: torch.tensor) -> torch.tensor:
\
\
           
        pass

    @abc.abstractmethod
    def build_backbone(self, config):
\
\
           
        pass

    @abc.abstractmethod
    def build_loss(self, config):
\
\
           
        pass

    @abc.abstractmethod
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
\
\
           
        pass

    @abc.abstractmethod
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
\
\
           
        pass
