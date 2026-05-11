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


import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import torch
from .utils.slowfast.models.video_model_builder import ResNet as ResNetOri
from .utils.slowfast.config.defaults import get_cfg
from torch import nn
import random


random_select = True
no_time_pool = False

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='altfreezing')
class AltFreezingDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        cfg = get_cfg()
        cfg.merge_from_str(config_text)
        cfg.NUM_GPUS = 1
        cfg.TEST.BATCH_SIZE = 1
        cfg.TRAIN.BATCH_SIZE = 1
        cfg.DATA.NUM_FRAMES = config['clip_size']
        self.resnet = ResNetOri(cfg)
        if config['pretrained'] is not None:
            print(f"loading pretrained model from {config['pretrained']}")
            pretrained_weights = torch.load(config['pretrained'], map_location='cpu', encoding='latin1')
            modified_weights = {k.replace("resnet.", ""): v for k, v in pretrained_weights.items()}
                                           
            modified_weights["head.projection.weight"] = modified_weights["head.projection.weight"][:1, :]
            modified_weights["head.projection.bias"] = modified_weights["head.projection.bias"][:1]
                             
            self.resnet.load_state_dict(modified_weights, strict=True)

        self.conv_dict = self.find_conv_layers(self.resnet)
        print(f"1x3x3 Conv: {self.conv_dict['spatial']}\n3x1x1 Conv:{self.conv_dict['temporal']}")
        self.train_batch_cnt = 0

        self.loss_func = nn.BCELoss()                                                                                       

    def find_conv_layers(self, module, parent_name='', conv_dict=None):
        if conv_dict is None:
            conv_dict = {'temporal': [], 'spatial': []}

        for name, sub_module in module.named_children():
            full_name = f'{parent_name}.{name}' if parent_name else name

            if isinstance(sub_module, nn.Conv3d):
                if sub_module.kernel_size == (3, 1, 1):
                    conv_dict['temporal'].append(full_name)
                if sub_module.kernel_size == (1, 3, 3):
                    conv_dict['spatial'].append(full_name)
            else:
                self.find_conv_layers(sub_module, full_name, conv_dict)

        return conv_dict

    def alternate_mode(self, target_mode):
        for layer_name in self.conv_dict['temporal']:
            layer = dict(self.resnet.named_modules())[layer_name]
            layer.weight.requires_grad = target_mode == 'temporal'
            if layer.bias is not None:
                layer.bias.requires_grad = target_mode == 'temporal'

        for layer_name in self.conv_dict['spatial']:
            layer = dict(self.resnet.named_modules())[layer_name]
            layer.weight.requires_grad = target_mode == 'spatial'
            if layer.bias is not None:
                layer.bias.requires_grad = target_mode == 'spatial'


    def build_backbone(self, config):
        pass
    
    def build_loss(self, config):
                                   
        loss_class = LOSSFUNC[config['loss_func']]
        return loss_class()
    
    def features(self, data_dict: dict) -> torch.tensor:
        inputs = [data_dict['image'].permute(0,2,1,3,4)]
        pred = self.resnet(inputs)
        output = {"final_output": pred}

        return output["final_output"]

    def classifier(self, features: torch.tensor):
        pass
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label'].float()
        pred = pred_dict['cls'].view(-1)
        loss = self.loss_func(pred, label)
        return {'overall': loss}
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
                                        
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def forward(self, data_dict: dict, inference=False) -> dict:
                             
        prob = self.features(data_dict)
                                                   
        pred_dict = {'cls': prob, 'prob': prob, 'feat': prob}
        if not inference:
            if self.train_batch_cnt % (20 + 1) == 0:
                self.alternate_mode('spatial')
            elif self.train_batch_cnt % (20 + 1) == 1:
                self.alternate_mode('temporal')

            self.train_batch_cnt += 1
        
        return pred_dict
