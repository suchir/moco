# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import torchvision.transforms as transforms

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MiniImagenet(Dataset):
    SPLIT_TO_FILE = {
        'train': 'miniImageNet_category_split_train_phase_train.pickle',
        'val': 'miniImageNet_category_split_val.pickle',
        'test': 'miniImageNet_category_split_test.pickle'
    }

    def __init__(self, *, split, transform):
        f = self.SPLIT_TO_FILE[split]
        with open(os.path.join('data', f), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        self.images = data['data']
        unique_labels = sorted(set(data['labels']))
        label_map = {x: i for i, x in enumerate(unique_labels)}
        self.labels = [label_map[x] for x in data['labels']]

        self.n_classes = len(unique_labels)
        self.class_to_images = [[] for _ in range(self.n_classes)]
        for i, label in enumerate(self.labels):
            self.class_to_images[label].append(i)

        self.transform = transforms.Compose([transforms.ToPILImage(), transform])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.transform(self.images[idx]), torch.tensor(self.labels[idx])
