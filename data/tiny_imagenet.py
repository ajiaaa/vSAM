from torchvision import models, utils, datasets, transforms
import torchvision
import numpy as np
import sys
import os
from PIL import Image
from utility.cutout import Cutout
import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

import glob
import os
from shutil import move
from os import rmdir
from torchvision.transforms.functional import InterpolationMode


class TinyImageNet:
    def __init__(self, batch_size, threads):

        mean, std = np.array([0.4802, 0.4481, 0.3975]), np.array([0.2770, 0.2691, 0.2821])
        train_transform = transforms.Compose([
            transforms.Resize(224),
            #transforms.RandomResizedCrop(32),
            #torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            #Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.Resize(224),
            #transforms.Resize((96, 96)),
            #transforms.RandomResizedCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        data_dir = r"E:\Datasets\tiny-imagenet-200\tiny-imagenet-200"
        train_set = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
        test_set = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=test_transform)

        train_sampler = RandomSampler(train_set)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=threads, sampler=train_sampler)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

