# original code: https://github.com/frank-xwang/RIDE-LongTailRecognition/blob/main/data_loader/imagenet_lt_data_loaders.py


import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import random
class LT_Dataset(Dataset):

    def __init__(self, root, txt, transform=None, use_randaug=False):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.use_randaug = use_randaug
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels  # Sampler needs to use targets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.use_randaug:
            r = random.random()
            if r < 0.5:
                sample = self.transform[0](sample)
            else:
                sample = self.transform[1](sample)
        else:
            if self.transform is not None:
                sample = self.transform(sample)

        # return sample, label, path
        return sample, label
    
'''
# Desired class ratios
class_ratios = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]
num_classes = 10
class_synset_list = ['n01558993', 'n01855672', 'n01514668', 'n01608432', 'n02342885', 
    'n01518878', 'n01860187', 'n02356798', 'n02325366', 'n01614925']
'''
class Imb_Dataset(Dataset):
    def __init__(self, path, transform=None, use_randaug=False, datatype='train',
                 num_classes=10,
                 class_ratios=[1000, 900, 800, 700, 500, 350, 200, 100, 60, 20],
                 class_synset_list = ['n01558993', 'n01855672','n01514668','n01608432', 'n02342885', 
                                      'n01518878', 'n01860187', 'n02356798', 'n02325366', 'n01614925']):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.use_randaug = use_randaug
        #load data from path and class_synset_list based on class_ratios
        for i in range(num_classes):
            class_synset = class_synset_list[i]
            class_path = os.path.join(path, class_synset)
            class_files = os.listdir(class_path)

            #randomly sample class_ratios[i] number of files from class_path with seed 42   
            random.seed(42)
            class_ratio = class_ratios[i]
            train_class_files = random.sample(class_files, class_ratio) 
            val_class_files = random.sample(list(set(class_files) - set(train_class_files)), int(class_ratio*0.2))
            if datatype == 'train':
                class_files = train_class_files
            else: #validation
                class_files = val_class_files
            for file in class_files:
                self.img_path.append(os.path.join(class_path, file))
                self.labels.append(i)
        self.targets = self.labels  # Sampler needs to use targets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.use_randaug:
            r = random.random()
            if r < 0.5:
                sample = self.transform[0](sample)
            else:
                sample = self.transform[1](sample)
        else:
            if self.transform is not None:
                sample = self.transform(sample)

        # return sample, label, path
        return sample, label