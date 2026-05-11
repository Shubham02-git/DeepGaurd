import torch.nn as nn
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="capsule_loss")
class CapsuleLoss(AbstractLossClass):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

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
           
                                  
        loss_t = self.cross_entropy_loss(inputs[:,0,:], targets)

        for i in range(inputs.size(1) - 1):
            loss_t = loss_t + self.cross_entropy_loss(inputs[:,i+1,:], targets)
        return loss_t
