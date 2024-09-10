import numpy as np
import glob
import torch.utils.data
import os
import math
from skimage import io, transform
from PIL import Image
import torch
import torchvision as vision
from torchvision import transforms, datasets
import random
from RandAugment.augmentations import *
from RandAugment.augmentations import Lighting, RandAugment

def encode_label(class_name, gender_classnames, ethnic_classnames):
    label = class_name.split('_')
    # print(label)
    gender_label = []
    gender_idx = gender_classnames.index(label[1])
    gender_label.append(gender_idx)
    
    ethnic_label = []
    ethnic_idx = ethnic_classnames.index(label[0])
    ethnic_label.append(ethnic_idx)
    
    return gender_label, ethnic_label
class MultiviewImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=12, shuffle=True):
        self.ethnic_classnames = ['han', 'wei']
        self.gender_classnames = ['man', 'woman']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        
        set_ = root_dir.split('/')[-1]
        
        parent_dir = root_dir.rsplit('/', 2)[0]
        self.filepaths = []
        
        img_classes=os.listdir(parent_dir)
        
        for i in range(len(img_classes)):
           
            all_files = sorted(glob.glob(parent_dir+'/'+img_classes[i]+'/'+set_+'/*.png'))
            # Select subset for different number of views
            stride = int(12/self.num_views) 
            
            all_files = all_files[::stride]
            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])
        if shuffle==True:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths)/num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i]*num_views:(rand_idx[i]+1)*num_views])
            self.filepaths = filepaths_new


        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])    
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            
    def __len__(self):
        return int(len(self.filepaths)/self.num_views)


    def __getitem__(self, idx):
        path = self.filepaths[idx*self.num_views]
        
        class_name = path.split('/')[-2]
        
        gender_label, ethnic_label = encode_label(class_name, self.gender_classnames, self.ethnic_classnames)
        label = {
            'gender_label':gender_label,
            'ethnic_label':ethnic_label
        }

        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            
            im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)
        
        return (label, torch.stack(imgs), self.filepaths[idx * self.num_views:(idx + 1) * self.num_views])



class SingleImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=12):
        self.ethnic_classnames = ['han', 'wei']
        self.gender_classnames = ['man', 'woman']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        
        set_ = root_dir.split('/')[-1]
        
        parent_dir = root_dir.rsplit('/',2)[0] 
        self.filepaths = []
        
        img_classes = os.listdir(parent_dir)
        for i in range(len(img_classes)):
           
            all_files = sorted(glob.glob(parent_dir+'/'+img_classes[i]+'/'+set_+'/*.png'))
            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])
        
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
       


    def __len__(self):
        return len(self.filepaths)


    def __getitem__(self, idx):
        path = self.filepaths[idx]
        
        class_name = path.split('/')[-2]

        gender_label, ethnic_label = encode_label(class_name, self.gender_classnames, self.ethnic_classnames)
        label = {
            'gender_label': gender_label,
            'ethnic_label': ethnic_label
        }
        
        im = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            im = self.transform(im)

        return (label, im, path)

