import sys

from torch import nn

sys.path.append('.')

import yaml
import numpy as np
from copy import deepcopy
import random
import torch
from torch.utils import data
from torchvision.utils import save_image
from training.dataset import DeepfakeAbstractBaseDataset
from einops import rearrange

FFpp_pool = ['FaceForensics++', 'FaceShifter', 'DeepFakeDetection', 'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT']   


def all_in_pool(inputs, pool):
    for each in inputs:
        if each not in pool:
            return False
    return True


class TALLDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
\
\
\
\
\
\
\
\
           
        super().__init__(config, mode)

        assert self.video_level, "TALL is a videl-based method"
        assert int(self.clip_size ** 0.5) ** 2 == self.clip_size, 'clip_size must be square of an integer, e.g., 4'

    def __getitem__(self, index, no_norm=False):
\
\
\
\
\
\
\
\
\
           
                                       
        image_paths = self.data_dict['image'][index]
        label = self.data_dict['label'][index]

        if not isinstance(image_paths, list):
            image_paths = [image_paths]                                                  

        image_tensors = []
        landmark_tensors = []
        mask_tensors = []
        augmentation_seed = None

        for image_path in image_paths:
                                                                                    
            if self.video_level and image_path == image_paths[0]:
                augmentation_seed = random.randint(0, 2 ** 32 - 1)

                                             
            mask_path = image_path.replace('frames', 'masks')                     
            landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')                         

                            
            try:
                image = self.load_rgb(image_path)
            except Exception as e:
                                                          
                print(f"Error loading image at index {index}: {e}")
                return self.__getitem__(0)
            image = np.array(image)                                                

                                                
            if self.config['with_mask']:
                mask = self.load_mask(mask_path)
            else:
                mask = None
            if self.config['with_landmark']:
                landmarks = self.load_landmark(landmark_path)
            else:
                landmarks = None

                                  
            if self.mode == 'train' and self.config['use_data_augmentation']:
                image_trans, landmarks_trans, mask_trans = self.data_aug(image, landmarks, mask, augmentation_seed)
            else:
                image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)

                                     
            if not no_norm:
                image_trans = self.normalize(self.to_tensor(image_trans))
                if self.config['with_landmark']:
                    landmarks_trans = torch.from_numpy(landmarks)
                if self.config['with_mask']:
                    mask_trans = torch.from_numpy(mask_trans)

            image_tensors.append(image_trans)
            landmark_tensors.append(landmarks_trans)
            mask_tensors.append(mask_trans)

        if self.video_level:

                                                              
            image_tensors = torch.stack(image_tensors, dim=0)

                                 
            F, C, H, W = image_tensors.shape
            x, y = np.random.randint(W), np.random.randint(H)
            x1 = np.clip(x - self.config['mask_grid_size'] // 2, 0, W)
            x2 = np.clip(x + self.config['mask_grid_size'] // 2, 0, W)
            y1 = np.clip(y - self.config['mask_grid_size'] // 2, 0, H)
            y2 = np.clip(y + self.config['mask_grid_size'] // 2, 0, H)
            image_tensors[:, :, y1:y2, x1:x2] = -1

                                                           
                                                             
                                                                                                     
                                                                                   
                                                                                                                    
                                                                                                        
                                                                          
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in
                       landmark_tensors):
                landmark_tensors = torch.stack(landmark_tensors, dim=0)
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = torch.stack(mask_tensors, dim=0)
        else:
                                        
            image_tensors = image_tensors[0]
                                                     
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in
                       landmark_tensors):
                landmark_tensors = landmark_tensors[0]
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = mask_tensors[0]

        return image_tensors, label, landmark_tensors, mask_tensors


if __name__ == "__main__":
    with open('training/config/detector/tall.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_set = TALLDataset(
        config=config,
        mode='train',
    )
    train_data_loader =\
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True,
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm

    for iteration, batch in enumerate(tqdm(train_data_loader)):
        print(batch['image'].shape)
        print(batch['label'])
        b, f, c, h, w = batch['image'].shape
        for i in range(f):
            img_tensor = batch['image'][0][i]
            img_tensor = img_tensor * torch.tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1) + torch.tensor(
                [0.5, 0.5, 0.5]).reshape(-1, 1, 1)
            save_image(img_tensor, f'{i}.png')

        break
