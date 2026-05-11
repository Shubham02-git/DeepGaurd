from enum import Enum
from functools import reduce

import cv2
import numpy as np
from scipy.ndimage import binary_dilation

from .DeepFakeMask import Mask


def dist(a, b):
    x1, y1 = a
    x2, y2 = b
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                                


def get_five_key(landmarks_68):
                                                    
    leye_center = (landmarks_68[36] + landmarks_68[39]) * 0.5
    reye_center = (landmarks_68[42] + landmarks_68[45]) * 0.5
    nose = landmarks_68[33]
    lmouth = landmarks_68[48]
    rmouth = landmarks_68[54]
    leye_left = landmarks_68[36]
    leye_right = landmarks_68[39]
    reye_left = landmarks_68[42]
    reye_right = landmarks_68[45]
    out = [
        tuple(x.astype("int32"))
        for x in [
            leye_center,
            reye_center,
            nose,
            lmouth,
            rmouth,
            leye_left,
            leye_right,
            reye_left,
            reye_right,
        ]
    ]
    return out


def remove_eyes(image, landmarks, opt):
                                            
    if opt == "l":
        (x1, y1), (x2, y2) = landmarks[5:7]
    elif opt == "r":
        (x1, y1), (x2, y2) = landmarks[7:9]
    elif opt == "b":
        (x1, y1), (x2, y2) = landmarks[:2]
    else:
        print("wrong region")
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    if opt != "b":
        dilation *= 4
    line = binary_dilation(line, iterations=dilation)
    return line


def remove_nose(image, landmarks):
    (x1, y1), (x2, y2) = landmarks[:2]
    x3, y3 = landmarks[2]
    mask = np.zeros_like(image[..., 0])
    x4 = int((x1 + x2) / 2)
    y4 = int((y1 + y2) / 2)
    line = cv2.line(mask, (x3, y3), (x4, y4), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    return line


def remove_mouth(image, landmarks):
    (x1, y1), (x2, y2) = landmarks[3:5]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 3)
    line = binary_dilation(line, iterations=dilation)
    return line


class SladdRegion(Enum):
    left_eye = 0
    right_eye = 1
    nose = 2
    mouth = 3
                 
    both_eyes = left_eye + right_eye     


class SladdMasking(Mask):

                                                                           
                                           
    ALL_REGIONS = [
        SladdRegion.left_eye,
        SladdRegion.right_eye,
        SladdRegion.nose,
        SladdRegion.mouth,
    ]
    REGIONS = [
        [SladdRegion.left_eye],
        [SladdRegion.right_eye],
        [SladdRegion.nose],
        [SladdRegion.mouth],
        [SladdRegion.left_eye, SladdRegion.right_eye],
        [SladdRegion.left_eye, SladdRegion.nose],
        [SladdRegion.right_eye, SladdRegion.nose],
        [SladdRegion.nose, SladdRegion.mouth],
        [SladdRegion.left_eye, SladdRegion.right_eye, SladdRegion.nose],
        ALL_REGIONS,
    ]

    def init(self, compose: bool = False, single: bool = True, **kwargs):
                                    
        self.compose = compose
        if compose:
            self.regions = SladdMasking.REGIONS
        else:
            self.regions = [reg for reg in SladdMasking.REGIONS if len(reg) == 1]
        if single:
            self.regions = [self.ALL_REGIONS]

    @property
    def total(self) -> int:
        return len(self.regions)

    @staticmethod
    def parse(img, reg, landmarks) -> np.ndarray:
        five_key = get_five_key(landmarks)
        if reg is SladdRegion.left_eye:
            mask = remove_eyes(img, five_key, "l")
        elif reg is SladdRegion.right_eye:
            mask = remove_eyes(img, five_key, "r")
        elif reg is SladdRegion.nose:
            mask = remove_nose(img, five_key)
        elif reg is SladdRegion.mouth:
            mask = remove_mouth(img, five_key)
        else:
            raise ValueError("Invalid region")
                                   
                                                   
        return mask

    def build_mask(self) -> np.ndarray:
        self.init()
        h, w = self.face.shape[:2]
                                  
        regs = [self.regions[0][self.idx]]
                                  
                                              
        masks = [SladdMasking.parse(self.face, reg, self.landmarks) for reg in regs]
        mask = reduce(np.maximum, masks)
        mask = mask.reshape([mask.shape[0],mask.shape[1], 1])

        return mask
