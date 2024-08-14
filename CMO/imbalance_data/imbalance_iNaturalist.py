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
def load_imb_inaturalist(image_dir, transform_train, transform_val, long_tail=False, random_select=False):
    #load data from txt files
    train_images_list = []
    train_labels_list = []
    val_images_list = []
    val_labels_list = []
    # if txt file doesnt exist in current directory, create the txt file
    #print current path
    print("loading imbalanced iNaturalist data-------")
    print(os.getcwd())
    
    if not long_tail and not random_select:
        txt_file = 'D:/anita/Research/XAI_Oversample/CMO/imbalance_data/iNaturalist_imb_train.txt'  
    elif long_tail and not random_select:
        txt_file = 'D:/anita/Research/XAI_Oversample/CMO/imbalance_data/iNaturalist_lt_train.txt'  
    else:
        txt_file = 'D:/anita/Research/XAI_Oversample/CMO/imbalance_data/iNaturalist_lt_random_train.txt'
    #txt_file = '/home/tcvcs/XAI_Oversample/CMO/imbalance_data/iNaturalist_imb_train.txt'    
    if not os.path.exists(txt_file):
        train_images_list, train_labels_list, val_images_list, val_labels_list = create_imblanced_inat_txt(image_dir, long_tail, random_select)
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

def calculate_class_numbers(range):
    start, end = range.split('-')
    return int(end) - int(start) + 1

# Create the imbalanced dataset and save the image directory to txt files
def create_imblanced_inat_txt(image_dir, long_tail, random_select=False):
    class_synset_list = ['00680-00705', '01962-01979', '03594-03607','04756-04764', '01392-01398', '01703-01707',  '04416-04419', '04819-04822', '04416-04419', '04739-04740']  
    train_images_list = []
    train_labels_list = []
    val_images_list = []
    val_labels_list = []
    
    class_numbers_in_each_range = [calculate_class_numbers(item) for item in class_synset_list] #[26, 18, 14, 9, 7, 5, 4, 4, 4, 2]
    print(class_numbers_in_each_range)
    
    if not long_tail:
        num_classes = 10
        class_ratios = [2000, 1500, 1000, 800, 650, 500, 350, 200, 120, 60]
    else:
        num_classes = sum(class_numbers_in_each_range)  #93
        class_ratios_ori = [120,100,80,60,40,30,20,10,5,5]   
        class_ratios = []
        for diff, ratio in zip(class_numbers_in_each_range, class_ratios_ori):
            class_ratios.extend([ratio] * diff)
            
    if not random_select:  
        j = 0            
        for i in range(10):
            class_synset = class_synset_list[i]
            start, end = class_synset.split('-')
            start, end = int(start), int(end)
            matched_dirs = []       
            if not long_tail: #sample the images within given range into the same image class.
                for item in os.listdir(image_dir):
                    set_num = int(item.split('_')[0])
                    if set_num >= start and set_num <= end:
                        matched_dirs.append(item)
                        
                all_images_in_class = []
                for directory in matched_dirs:
                    for root, dirs, files in os.walk(os.path.join(image_dir, directory)):
                        for file in files:
                            all_images_in_class.append(os.path.join(root, file))
                        
                sample_number = class_ratios[i]
                
                if sample_number < len(all_images_in_class):
                    train_class_files = random.sample(all_images_in_class, sample_number)
                    val_class_files = random.sample(list(set(all_images_in_class) - set(train_class_files)), 60)
                else:
                    train_class_files = all_images_in_class
                    val_class_files = random.sample(all_images_in_class, 60)
                
                for file in train_class_files:
                    train_images_list.append(file)
                    train_labels_list.append(i)
                for file in val_class_files:
                    val_images_list.append(file)
                    val_labels_list.append(i)
                    
            else: #long tail, each directory is a class, in total 93 classes instead of 10 classes.
                #j = class_numbers_in_each_range[i]
                for item in os.listdir(image_dir):
                    set_num = int(item.split('_')[0])
                    #if the set_num is within the range, this directory is a class
                    if set_num >= start and set_num <= end:
                        all_images_in_class = []
                        for root, dirs, files in os.walk(os.path.join(image_dir, item)):
                            for file in files:
                                all_images_in_class.append(os.path.join(root, file))
                                
                        sample_number = class_ratios[j]
                        
                        if sample_number < len(all_images_in_class):
                            train_class_files = random.sample(all_images_in_class, sample_number)
                            remain = list(set(all_images_in_class) - set(train_class_files))
                            if len(remain)>=60:
                                val_class_files = random.sample(remain, 60)
                            else:
                                val_class_files = random.sample(all_images_in_class, 60)
                        else:
                            train_class_files = all_images_in_class
                            val_class_files = random.sample(all_images_in_class, 60)
                        
                        for file in train_class_files:
                            train_images_list.append(file)
                            train_labels_list.append(j)
                        for file in val_class_files:
                            val_images_list.append(file)
                            val_labels_list.append(j)   
                        j = j + 1
    
    else: #TODO: randomly select images from the whole dataset to create 91 classes dataset: random_lt_iNaturalist  
        if not long_tail:
            print('Random selection is only for long tail dataset')
            return  #exit
        
        total_classes = 93 
        #randomly select 91 directory from image_dir
        all_dirs = os.listdir(image_dir)
        selected_dirs = random.sample(all_dirs, total_classes)
        for i in range(total_classes):
            all_images_in_class = []
            for root, dirs, files in os.walk(os.path.join(image_dir, selected_dirs[i])):
                for file in files:
                    all_images_in_class.append(os.path.join(root, file))
                    
            sample_number = class_ratios[i]
            if sample_number < len(all_images_in_class):
                train_class_files = random.sample(all_images_in_class, sample_number)
                remain = list(set(all_images_in_class) - set(train_class_files))
                if len(remain)>=60:
                    val_class_files = random.sample(remain, 60)
                else:
                    val_class_files = random.sample(all_images_in_class, 60)
            else:
                train_class_files = all_images_in_class
                val_class_files = random.sample(all_images_in_class, 60)
            
            for file in train_class_files:
                train_images_list.append(file)
                train_labels_list.append(i)
            for file in val_class_files:
                val_images_list.append(file)
                val_labels_list.append(i)
            
    # Save the imbalanced dataset to text files
    if not long_tail and not random_select:
        train_file_name = 'D:/anita/Research/XAI_Oversample/CMO/imbalance_data/iNaturalist_imb_train.txt'
        val_file_name = 'D:/anita/Research/XAI_Oversample/CMO/imbalance_data/iNaturalist_imb_val.txt'
    elif long_tail and not random_select:
        train_file_name = 'D:/anita/Research/XAI_Oversample/CMO/imbalance_data/iNaturalist_lt_train.txt'
        val_file_name = 'D:/anita/Research/XAI_Oversample/CMO/imbalance_data/iNaturalist_lt_val.txt'
    else:
        train_file_name = 'D:/anita/Research/XAI_Oversample/CMO/imbalance_data/iNaturalist_lt_random_train.txt'
        val_file_name = 'D:/anita/Research/XAI_Oversample/CMO/imbalance_data/iNaturalist_lt_random_val.txt'
        
    with open(train_file_name, 'w') as img_file:
        for i in range(len(train_images_list)):
            #write image path and label in one line: path space label
            img_file.write(train_images_list[i] + ' ' + str(train_labels_list[i]) + '\n')

    with open(val_file_name, 'w') as img_file:
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
train_images_list, train_labels_list, val_images_list, val_labels_list = load_imb_inaturalist(img_dir, transform_train, transform_val, True, True)
'''
