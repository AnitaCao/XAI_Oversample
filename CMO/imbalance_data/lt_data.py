# original code: https://github.com/frank-xwang/RIDE-LongTailRecognition/blob/main/data_loader/imagenet_lt_data_loaders.py


import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import random
import torchvision.transforms as transforms

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
    def __init__(self, img_dir, img_path_list, labels_list, transform=None, use_randaug=False, datatype='train',
                 num_classes=10,
                 class_ratios=[1000, 900, 800, 700, 500, 350, 200, 100, 60, 20],
                 class_synset_list = ['n01558993', 'n01855672','n01514668','n01608432', 'n02342885', 
                                      'n01518878', 'n01860187', 'n02356798', 'n02325366', 'n01614925']):   
        self.transform = transform
        self.use_randaug = use_randaug
        self.img_path = []
        self.labels = []

        if img_path_list is not None and labels_list is not None:
            # create imbalanced dataset from given img_path_list and labels_list. 
            # (Fixed class_ratios, class_synset_list and images.)
            self.img_path = img_path_list
            self.labels = labels_list
        else:
            # create imbalanced dataset from given directory and class_ratios and class_synset_list 
            # (randomly sample files from each class based on class_ratios)
            self.img_path = []
            self.labels = []
            #load data from path and class_synset_list based on class_ratios
            for i in range(num_classes):
                class_synset = class_synset_list[i]
                class_path = os.path.join(img_dir, class_synset)
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
    

# Load the imbalanced dataset from the given txt files
def load_imb_imagenet(image_dir, transform_train, transfrom_val, use_randaug=False):
    #load data from txt files
    train_images_list = []
    train_labels_list = []
    val_images_list = []
    val_labels_list = []
    # if txt file doesnt exist in current directory, create the txt file
    if not os.path.exists('imbalance_data/ImageNet_imb_train.txt'):
        train_images_list, train_labels_list, val_images_list, val_labels_list = create_imblanced_imagenet_txt(image_dir)
    else:
        with open('imbalance_data/ImageNet_imb_train.txt') as f:
            for line in f:
                train_images_list.append(os.path.join(image_dir, line.split()[0]))
                train_labels_list.append(int(line.split()[1]))
        with open('imbalance_data/ImageNet_imb_val.txt') as f:
            for line in f:
                val_images_list.append(os.path.join(image_dir, line.split()[0]))
                val_labels_list.append(int(line.split()[1]))
    
    train_dataset = Imb_Dataset(image_dir,train_images_list, train_labels_list,transform_train, use_randaug)
    val_dataset = Imb_Dataset(image_dir,val_images_list, val_labels_list,transfrom_val, use_randaug)

    return train_dataset, val_dataset

# Create the imbalanced dataset and save the image directory to txt files
def create_imblanced_imagenet_txt(image_dir):  
    num_classes = 10
    class_ratios = [1000, 950, 850, 750, 550, 350, 250, 150, 80, 40]
    class_synset_list = ['n01558993', 'n01855672','n01514668','n01608432', 'n02342885', 
                                      'n01518878', 'n01860187', 'n02356798', 'n02325366', 'n01614925']

    train_images_list = []
    train_labels_list = []
    val_images_list = []
    val_labels_list = []

    # if txt file doesnt exist, create the txt file
    for i in range(num_classes):
        class_synset = class_synset_list[i]
        class_path = os.path.join(image_dir, class_synset)
        class_files = os.listdir(class_path)
        train_class_files = random.sample(class_files, class_ratios[i]) 
        val_class_files = random.sample(list(set(class_files) - set(train_class_files)), int(class_ratios[i]*0.2))
        for file in train_class_files:
            train_images_list.append(os.path.join(class_path, file))
            train_labels_list.append(i)
        for file in val_class_files:
            val_images_list.append(os.path.join(class_path, file))
            val_labels_list.append(i)

    # Save the imbalanced dataset to text files
    with open('imbalance_data/ImageNet_imb_train.txt', 'w') as img_file:
        for i in range(len(train_images_list)):
            #write image path and label in one line: path space label
            img_file.write(train_images_list[i] + ' ' + str(train_labels_list[i]) + '\n')

    with open('imbalance_data/ImageNet_imb_val.txt', 'w') as img_file:
        for i in range(len(val_images_list)):
            img_file.write(val_images_list[i] + ' ' + str(val_labels_list[i]) + '\n')
    
    return train_images_list, train_labels_list, val_images_list, val_labels_list

'''
#Testing
img_dir ='D:/anita/Research/competitions/imagenet-object-localization-challenge/ILSVRC/ILSVRC/Data/CLS-LOC/train/'
transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
train_dataset, val_dataset = load_imb_imagenet(img_dir, transform_train, transform_val)
'''