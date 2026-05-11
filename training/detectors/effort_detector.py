import os
import math
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

import loralib as lora
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='effort')
class EffortDetector(nn.Module):
    def __init__(self, config=None):
        super(EffortDetector, self).__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = nn.Linear(1024, 2)
        self.loss_func = nn.CrossEntropyLoss()
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

    def build_backbone(self, config):
                        
                                                              
        
                                                   
                                                   
        
                          
        clip_model = CLIPModel.from_pretrained("../../huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/")

                                            
                                  
        clip_model.vision_model = apply_svd_residual_to_self_attn(clip_model.vision_model, r=1024-1)

        for name, param in clip_model.vision_model.named_parameters():
            print('{}: {}'.format(name, param.requires_grad))
        num_param = sum(p.numel() for p in clip_model.vision_model.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in clip_model.vision_model.parameters())
        print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))

        return clip_model.vision_model

    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])['pooler_output']
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

                                                                     
                                    
                                 
                                            
        
                               
                          
                                
                                                
                                                       
                                                                                        
                                                                            
        
                               
                                               
                                           
        
                                       
                          

    def compute_weight_loss(self):
        weight_sum_dict = {}
        num_weight_dict = {}
        for name, module in self.backbone.named_modules():
            if isinstance(module, SVDResidualLinear):
                weight_curr = module.compute_current_weight()
                if str(weight_curr.size()) not in weight_sum_dict.keys():
                    weight_sum_dict[str(weight_curr.size())] = weight_curr
                    num_weight_dict[str(weight_curr.size())] = 1
                else:
                    weight_sum_dict[str(weight_curr.size())] += weight_curr
                    num_weight_dict[str(weight_curr.size())] += 1
        
        loss2 = 0.0
        for k in weight_sum_dict.keys():
            _, S_sum, _ = torch.linalg.svd(weight_sum_dict[k], full_matrices=False)
            loss2 += -torch.mean(S_sum)
        loss2 /= len(weight_sum_dict.keys())
        return loss2

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']                                
        pred = pred_dict['cls']                                                

                                                
        loss = self.loss_func(pred, label)

                                                
        mask_real = label == 0                  
        mask_fake = label == 1                  

                                     
        if mask_real.sum() > 0:
            pred_real = pred[mask_real]
            label_real = label[mask_real]
            loss_real = self.loss_func(pred_real, label_real)
        else:
                                      
            loss_real = torch.tensor(0.0, device=pred.device)

                                     
        if mask_fake.sum() > 0:
            pred_fake = pred[mask_fake]
            label_fake = label[mask_fake]
            loss_fake = self.loss_func(pred_fake, label_fake)
        else:
                                      
            loss_fake = torch.tensor(0.0, device=pred.device)
        

                                            
                                     

                                             
        loss_dict = {
            'overall': loss,
            'real_loss': loss_real,
            'fake_loss': loss_fake,
                                 
        }
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
                                        
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
                                      
        features = self.features(data_dict)
                                          
        pred = self.classifier(features)
                                         
        prob = torch.softmax(pred, dim=1)[:, 1]
                                                   
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}

        return pred_dict


                                                              
class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r                                            

                                  
        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

              
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def compute_current_weight(self):
        if self.S_residual is not None:
            return self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        else:
            return self.weight_main

    def forward(self, x):
        if hasattr(self, 'U_residual') and hasattr(self, 'V_residual') and self.S_residual is not None:
                                             
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
                                                                     
            weight = self.weight_main + residual_weight
        else:
                                                                          
            weight = self.weight_main

        return F.linear(x, weight, self.bias)
    
    def compute_orthogonal_loss(self):
                                                                      
        UUT_residual = self.U_residual @ self.U_residual.t()
        VVT_residual = self.V_residual @ self.V_residual.t()
        
                                      
        UUT_residual_identity = torch.eye(UUT_residual.size(0), device=UUT_residual.device)
        VVT_residual_identity = torch.eye(VVT_residual.size(0), device=VVT_residual.device)
        
                        
        loss = 0.5 * torch.norm(UUT_residual - UUT_residual_identity, p='fro') + 0.5 * torch.norm(VVT_residual - VVT_residual_identity, p='fro')
        
        return loss
        

                                                                                       
def apply_svd_residual_to_self_attn(model, r):
    for name, module in model.named_children():
        if 'self_attn' in name:
                                                     
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                                                        
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                                                                        
                    setattr(parent_module, sub_module_names[-1], replace_with_svd_residual(sub_module, r))
        else:
                                                
            apply_svd_residual_to_self_attn(module, r)
                                                                
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


                                                     
def replace_with_svd_residual(module, r):
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

                                         
        new_module = SVDResidualLinear(in_features, out_features, r, bias=bias, init_weight=module.weight.data.clone())

        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)

                                            
        U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)

                                                            
        r = min(r, len(S))                                                          

                                                      
        U_r = U[:, :r]                                
        S_r = S[:r]                      
        Vh_r = Vh[:r, :]                             

                                             
        weight_main = U_r @ torch.diag(S_r) @ Vh_r

                             
        new_module.weight_main.data.copy_(weight_main)

                                         
        U_residual = U[:, r:]                                  
        S_residual = S[r:]                        
        Vh_residual = Vh[r:, :]                               

        if len(S_residual) > 0:
                                     
            new_module.S_residual = nn.Parameter(S_residual.clone())
                                                          
            new_module.U_residual = nn.Parameter(U_residual.clone())
            new_module.V_residual = nn.Parameter(Vh_residual.clone())
        else:
                                                         
            new_module.S_residual = None
            new_module.U_residual = None
            new_module.V_residual = None

        return new_module
    else:
        return module
