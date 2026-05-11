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
from loss import LOSSFUNC

import loralib as lora

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='videomae')
class VideoMAEDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.fc_norm = nn.LayerNorm(768)
        self.head = nn.Linear(768, 2)
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        from transformers import VideoMAEModel
        backbone = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        return backbone

    
    def build_loss(self, config):
                                   
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.Tensor:
                                                  
                                                               
                                    
                                                                    
                                                                                                      
                                    
                                                                                        

        outputs = self.backbone(data_dict['image'], output_hidden_states=True)
        sequence_output = outputs[0]
        video_level_features = self.fc_norm(sequence_output.mean(1))
        return video_level_features

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        return {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
                                        
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
                                                              
        self.video_names = []
        return metric_batch_dict
    
    def forward(self, data_dict: dict, inference=False) -> dict:
                                      
        features = self.features(data_dict)
                                          
        pred = self.classifier(features)
                                         
        prob = torch.softmax(pred, dim=1)[:, 1]
                                                   
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        
        return pred_dict
