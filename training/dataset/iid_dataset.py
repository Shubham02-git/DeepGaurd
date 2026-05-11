import os.path
from copy import deepcopy
import cv2
import math
import torch
import random

import yaml
from PIL import Image, ImageDraw
import numpy as np
from torch.utils.data import DataLoader

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

class IIDDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)


    def __getitem__(self, index):
                                       
        image_path = self.data_dict['image'][index]
        if '\\' in image_path:
            per = image_path.split('\\')[-2]
        else:
            per = image_path.split('/')[-2]
        id_index = int(per.split('_')[-1])                 
        label = self.data_dict['label'][index]

                        
        try:
            image = self.load_rgb(image_path)
        except Exception as e:
                                                      
            print(f"Error loading image at index {index}: {e}")
            return self.__getitem__(0)
        image = np.array(image)                                                

                              
        image_trans,_,_ = self.data_aug(image)

                                 
        image_trans = self.normalize(self.to_tensor(image_trans))

        return id_index, image_trans, label

    @staticmethod
    def collate_fn(batch):
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
           
                                                               
        id_indexes, image_trans, label = zip(*batch)

                                                            
        images = torch.stack(image_trans, dim=0)
        labels = torch.LongTensor(label)
        ids = torch.LongTensor(id_indexes)
                                            
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['id_index'] = ids
        data_dict['mask']=None
        data_dict['landmark']=None
        return data_dict


def draw_landmark(img,landmark):
    draw = ImageDraw.Draw(img)

                                                             
                    
             
    for i, point in enumerate(landmark):
                   
        draw.ellipse((point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1), fill=(255, 0, 0))
                    
        draw.text((point[0], point[1]), str(i), fill=(255, 255, 255))
    return img


if __name__ == '__main__':
    detector_path = r"./training/config/detector/xception.yaml"
                                                          
    with open(detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config2['data_manner'] = 'lmdb'
    config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    config.update(config2)
    dataset = IIDDataset(config=config)
    batch_size = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,collate_fn=dataset.collate_fn)

    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: {batch}")

                                                  
        img = batch['img']
