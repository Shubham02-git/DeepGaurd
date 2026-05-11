                      
                                                                       

                            

import torch
import torch.nn as nn

class ResNetBasicHead(nn.Module):
\
\
\
\
\
\
       

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
    ):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
                                                                          
                                                                             
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

                                             
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
                                             
        x = x.permute((0, 2, 3, 4, 1))
                          
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

                                                
                               
                                   
        x = self.act(x)
        x = x.view(x.shape[0], -1)
        return x
