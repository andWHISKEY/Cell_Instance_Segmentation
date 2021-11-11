import numpy as np
import torch
import cv2
from torchvision.transforms.functional import rotate, vflip, hflip

class ToTensor_test(object):
    def __call__(self, x):
        image = x
        image = np.transpose(image, [2, 0, 1])
        image = torch.tensor(image)/255.
        return image

class Resize_test(object):
    def __init__(self, length):
        self.length = length
    def __call__(self, x):
        image = x
        image = cv2.resize(image, dsize=(self.length, self.length))
        return np.asarray(image)

class ToTensor(object):
    def __call__(self, x):
        image, mask = x
        mask = np.expand_dims(mask, axis=-1)
        image = np.transpose(image, [2, 0, 1])
        mask = np.transpose(mask, [2, 0, 1])
        image = torch.tensor(image)/255.
        mask = torch.tensor(mask)/1.

        return image, mask

class RandomFlip(object):
    def __call__(self, x):
        image, mask = x
        if np.random.random(1)>0.5:
            image = hflip(image)
            mask = hflip(mask)
        if np.random.random(1)>0.5:
            image = vflip(image)
            mask = vflip(mask)

        return image, mask

class Resize(object):
    def __init__(self, length):
        self.length = length
    def __call__(self, x):
        image, mask = x
        mask = np.expand_dims(mask, axis=-1)
        # mask = mask*255
        image = cv2.resize(image, dsize=(self.length, self.length))
        mask = cv2.resize(mask, dsize=(self.length, self.length))
        return np.asarray(image), np.asarray(mask)

