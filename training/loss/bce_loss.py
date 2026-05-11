import torch.nn as nn
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="bce")
class BCELoss(AbstractLossClass):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, inputs, targets):
\
\
\
\
\
\
\
\
\
           
                              
        loss = self.loss_fn(inputs, targets.float())

        return loss