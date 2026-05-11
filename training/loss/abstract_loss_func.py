import torch.nn as nn

class AbstractLossClass(nn.Module):
                                            
    def __init__(self):
        super(AbstractLossClass, self).__init__()

    def forward(self, pred, label):
\
\
\
\
\
\
\
           
        raise NotImplementedError('Each subclass should implement the forward method.')
