import os
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
import random

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='spsl')
class SpslDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
                              
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)

                                                                               
        state_dict = torch.load(config['pretrained'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}

                                      
        conv1_data = state_dict.pop('conv1.weight')

        backbone.load_state_dict(state_dict, False)
        logger.info('Load pretrained model from {}'.format(config['pretrained']))

                       
                                                            
        backbone.conv1 = nn.Conv2d(4, 32, 3, 2, 0, bias=False)
        avg_conv1_data = conv1_data.mean(dim=1, keepdim=True)                                   
        backbone.conv1.weight.data = avg_conv1_data.repeat(1, 4, 1, 1)                                                         
        logger.info('Copy conv1 from pretrained model')
        return backbone

    def build_loss(self, config):
                                   
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict, phase_fea) -> torch.Tensor:
        features = torch.cat((data_dict['image'], phase_fea), dim=1)
        return self.backbone.features(features)

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
                                        
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
                                                              
        self.video_names = []
        return metric_batch_dict
    
    def forward(self, data_dict: dict, inference=False) -> dict:
                                
        phase_fea = self.phase_without_amplitude(data_dict['image'])
            
        features = self.features(data_dict, phase_fea)
                                          
        pred = self.classifier(features)
                                         
        prob = torch.softmax(pred, dim=1)[:, 1]
                                                   
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}

        return pred_dict

    def phase_without_amplitude(self, img):
                              
        gray_img = torch.mean(img, dim=1, keepdim=True)                                   
                                             
        X = torch.fft.fftn(gray_img,dim=(-1,-2))
                                
                                                    
        phase_spectrum = torch.angle(X)
                                                                                     
        reconstructed_X = torch.exp(1j * phase_spectrum)
                                                         
        reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X,dim=(-1,-2)))
                                                                        
        return reconstructed_x
