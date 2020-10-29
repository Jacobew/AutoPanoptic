# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, seg_target=None):
        if seg_target is not None:
            assert image.size == seg_target.size, 'image size:{} and segmentation size:{} should match'.format(image.size, seg_target.size)
        for t in self.transforms:
            if seg_target is None:
                image, target = t(image, target)
            else:
                image, target, seg_target = t(image, target, seg_target)
        if seg_target is None:
            return image, target
        else:
            return image, target, seg_target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomResizewithCrop(object):
    def __init__(self, is_train):
        self.is_train = is_train
        self.WIDTH_AFTER_CROP = 1024
        self.HEIGHT_AFTER_CROP = 512

    def __call__(self, image, target, seg_target=None):
        # during testing, do not resize and crop
        if not self.is_train:
            if seg_target is None:
                return image, target
            else:
                return image, target, seg_target

        scale_ratio = np.random.uniform(0.5, 2)
        
        new_w = int(scale_ratio * w)
        new_h = int(scale_ratio * h)
        image = F.resize(image, (new_h, new_w))
        target = target.resize(image.size)
        if seg_target is not None:
            seg_target = F.resize(seg_target, (new_h, new_w))
        # random crop to 512 * 1024
        left_upper_x = np.random.randint(0, new_w-self.WIDTH_AFTER_CROP + 1)
        left_upper_y = np.random.randint(0, new_h-self.HEIGHT_AFTER_CROP + 1)
        box = (left_upper_x, left_upper_y, left_upper_x+self.WIDTH_AFTER_CROP, left_upper_y+self.HEIGHT_AFTER_CROP)

        image = image.crop(box)
        target = target.crop(box)
        if seg_target is not None:
            seg_target = seg_target.crop(box)
            return image, target, seg_target
        return image, target


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target, seg_target=None):
        if seg_target is not None:
            assert image.size == seg_target.size, 'image size:{} and segmentation size:{} should match'.format(image.size, seg_target.size)
        size = self.get_size(image.size)
        image = F.resize(image, size)
        target = target.resize(image.size)
        if seg_target is not None:
            seg_target = F.resize(seg_target, size, Image.NEAREST)
            return image, target, seg_target
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target, seg_target=None):
        if seg_target is not None:
            assert image.size == seg_target.size, 'image size:{} and segmentation size:{} should match'.format(image.size, seg_target.size)
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
            if seg_target is not None:
                seg_target = F.hflip(seg_target)
                
        if seg_target is not None:
            return image, target, seg_target
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target, seg_target=None):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
            if seg_target is not None:
                seg_target = F.vflip(seg_target)

        if seg_target is not None:
            return image, target, seg_target
        return image, target

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target, seg_target=None):
        image = self.color_jitter(image)
        if seg_target is not None:
            return image, target, seg_target
        return image, target


class ToTensor(object):
    def __call__(self, image, target, seg_target=None):
        if seg_target is not None:
            assert image.size == seg_target.size, 'image size:{} and segmentation size:{} should match'.format(image.size, seg_target.size)
        if seg_target is not None:
            return F.to_tensor(image), target, torch.from_numpy(np.array(seg_target))

        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target, seg_target=None):
        if seg_target is not None:
            assert image.shape[-2:] == seg_target.shape[-2:], 'image size:{} and segmentation size:{} should match'.format(image.size, seg_target.size)
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if seg_target is not None:
            return image, target, seg_target
        return image, target
