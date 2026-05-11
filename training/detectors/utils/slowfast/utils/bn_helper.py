                      
                                                                       

                

import itertools
import torch


@torch.no_grad()
def compute_and_update_bn_stats(model, data_loader, num_batches=200):
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
       

                                 
    bn_layers = [
        m
        for m in model.modules()
        if any(
            (
                isinstance(m, bn_type)
                for bn_type in (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.BatchNorm3d,
                )
            )
        )
    ]

                                                                            
                           
                                                                                
                                                                        
    momentum_actual = [bn.momentum for bn in bn_layers]
    for bn in bn_layers:
        bn.momentum = 1.0

                                                                      
    running_mean = [torch.zeros_like(bn.running_mean) for bn in bn_layers]
    running_square_mean = [torch.zeros_like(bn.running_var) for bn in bn_layers]

    for ind, (inputs, _, _) in enumerate(
        itertools.islice(data_loader, num_batches)
    ):
                                                    
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].float().cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        model(inputs)

        for i, bn in enumerate(bn_layers):
                                       
            running_mean[i] += (bn.running_mean - running_mean[i]) / (ind + 1)
                                         
            cur_square_mean = bn.running_var + bn.running_mean ** 2
            running_square_mean[i] += (
                cur_square_mean - running_square_mean[i]
            ) / (ind + 1)

    for i, bn in enumerate(bn_layers):
        bn.running_mean = running_mean[i]
                                     
        bn.running_var = running_square_mean[i] - bn.running_mean ** 2
                                    
        bn.momentum = momentum_actual[i]
