# original code: https://github.com/frank-xwang/RIDE-LongTailRecognition/blob/main/data_loader/imagenet_lt_data_loaders.py


import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import random
import torchvision.transforms as transforms

class iNat_Dataset(Dataset):

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
class_ratios = [2000, 1500, 1000, 800, 650, 500, 350, 200, 120, 60]
class_synset_list = ['00680-00705', '01962-01979', '03594-03607','04756-04764', 
    '01392-01398', '01703-01707',  '04416-04419', '04819-04822', '04416-04419', '04739-04740']
TODO: add class names
'''
class Imb_iNat_Dataset(Dataset):
    def __init__(self, img_dir, img_path_list, labels_list, transform=None, use_randaug=False, datatype='train'):    
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
            train_images_list, train_labels_list, val_images_list, val_labels_list = create_imblanced_inat_txt(img_dir)
            if datatype == 'train':
                self.img_path = train_images_list
                self.labels = train_labels_list
            else:
                self.img_path = val_images_list
                self.labels = val_labels_list      
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
def load_imb_inaturalist(image_dir, transform_train, transform_val, use_randaug=False):
    #load data from txt files
    train_images_list = []
    train_labels_list = []
    val_images_list = []
    val_labels_list = []
    # if txt file doesnt exist in current directory, create the txt file
    #print current path
    print("loading imbalanced iNaturalist data-------")
    print(os.getcwd())
    
    #txt_file = 'D:/anita/Research/XAI_Oversample/CMO/imbalance_data/iNaturalist_imb_train.txt' #local PC
    txt_file = '/home/tcvcs/XAI_Oversample/CMO/imbalance_data/iNaturalist_imb_train.txt'
    if not os.path.exists(txt_file):
        train_images_list, train_labels_list, val_images_list, val_labels_list = create_imblanced_inat_txt(image_dir)
    else:
        with open(txt_file) as f:
            for line in f:
                train_images_list.append(os.path.join(image_dir, line.split()[0]))
                train_labels_list.append(int(line.split()[1]))
        with open(txt_file) as f:
            for line in f:
                val_images_list.append(os.path.join(image_dir, line.split()[0]))
                val_labels_list.append(int(line.split()[1]))
    
    #train_dataset = Imb_iNat_Dataset(image_dir,train_images_list, train_labels_list,transform_train, use_randaug)
    #val_dataset = Imb_iNat_Dataset(image_dir,val_images_list, val_labels_list,transform_val, use_randaug)

    return train_images_list, train_labels_list, val_images_list, val_labels_list

# Create the imbalanced dataset and save the image directory to txt files
def create_imblanced_inat_txt(image_dir):  
    num_classes = 10
    class_ratios = [2000, 1500, 1000, 800, 650, 500, 350, 200, 120, 60]
    class_synset_list = ['00680-00705', '01962-01979', '03594-03607','04756-04764', '01392-01398', '01703-01707',  '04416-04419', '04819-04822', '04416-04419', '04739-04740']
    train_images_list = []
    train_labels_list = []
    val_images_list = []
    val_labels_list = []

    # if txt file doesnt exist, create the txt file
    for i in range(num_classes):
        class_synset = class_synset_list[i]
        
        #get the matched directories within class_synset range
        matched_dirs = []
        for item in os.listdir(image_dir):
            start, end = class_synset.split('-')
            start, end = int(start), int(end)
            set_num = int(item.split('_')[0])
            if set_num >= start and set_num <= end:
                matched_dirs.append(item)
        # get images from the matched directories based on class_ratios       
        all_images = []
        for directory in matched_dirs:
            for root, dirs, files in os.walk(os.path.join(image_dir, directory)):
                for file in files:
                    all_images.append(os.path.join(root, file))

        num_images = class_ratios[i]
        if num_images < len(all_images):
            train_class_files = random.sample(all_images, num_images)          
            val_class_files= random.sample(list(set(all_images) - set(train_class_files)), 60)       
        else:
            train_images_files = all_images
            val_class_files = random.sample(all_images, 60)
        
        for file in train_class_files:
            train_images_list.append(file)
            train_labels_list.append(i)
        for file in val_class_files:
            val_images_list.append(file)
            val_labels_list.append(i)

    
    # Save the imbalanced dataset to text files
    #txt_file_train = 'D:/anita/Research/XAI_Oversample/CMO/imbalance_data/iNaturalist_imb_train.txt' #local PC
    txt_file_train = '/home/tcvcs/XAI_Oversample/CMO/imbalance_data/iNaturalist_imb_train.txt' #RC account
    with open(txt_file, 'w') as img_file:
        for i in range(len(train_images_list)):
            #write image path and label in one line: path space label
            img_file.write(train_images_list[i] + ' ' + str(train_labels_list[i]) + '\n')
    
    #txt_file_val = 'D:/anita/Research/XAI_Oversample/CMO/imbalance_data/iNaturalist_imb_val.txt' #local PC
    txt_file_val = '/home/tcvcs/XAI_Oversample/CMO/imbalance_data/iNaturalist_imb_val.txt' #RC account
    with open(txt_file_val, 'w') as img_file:
        for i in range(len(val_images_list)):
            img_file.write(val_images_list[i] + ' ' + str(val_labels_list[i]) + '\n')
    
    return train_images_list, train_labels_list, val_images_list, val_labels_list

'''
#Testing
img_dir ='D:/anita/Research/iNaturalist/train/'
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
train_dataset, val_dataset = load_imb_inaturalist(img_dir, transform_train, transform_val)
'''
