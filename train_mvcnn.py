import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse
import torch.nn.functional as F


from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN

# 参数定义
parser = argparse.ArgumentParser()
# add_argument() 方法，该方法用于指定程序能够接受哪些命令行选项
# 当'-'和'--'同时出现的时候，系统默认后者为参数名，前者不是，但是在命令行输入的时候没有这个区分
# argparse默认的变量名是--或-后面的字符串，但是你也可以通过dest=xxx来设置参数的变量名，然后在代码中用args.xxx来获取参数的值。
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="convnext-001")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=2)# it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=1000)
parser.add_argument("-lr", type=float, help="learning rate", default=0.001)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.01)
# 当action这一选项存在时，为 args.no_pretraining 赋值为 True。没有指定时则隐含地赋值为 False。
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="convnext") 
# 视角数
parser.add_argument("-num_views", type=int, help="number of views", default=12)
# change path
parser.add_argument("-train_path", type=str, default="F:/Pythonproject/project1/data_augment/multiview_skulldata1/*/train1")
parser.add_argument("-val_path", type=str, default="F:/Pythonproject/project1/data_augment/multiview_skulldata1/*/test")
# set_defaults()可以设置一些参数的默认值
parser.set_defaults(train=False)

class CustomSubset1(SingleImgDataset, torch.utils.data.Subset):
    def __init__(self, dataset, indices):
        super().__init__(root_dir=dataset.root_dir, scale_aug=dataset.scale_aug, rot_aug=dataset.rot_aug,
                         test_mode=dataset.test_mode)

        self.indices = indices

    def __getitem__(self, idx):
        idx = self.indices[idx]
        # filepaths = self.filepaths[idx]
        # labels = self.la
        # print(idx)
        return super().__getitem__(idx)

    def __len__(self):
        return len(self.indices)
class CustomSubset2(MultiviewImgDataset, torch.utils.data.Subset):
    def __init__(self, dataset, indices):
        super().__init__(root_dir=dataset.root_dir, scale_aug=dataset.scale_aug, rot_aug=dataset.rot_aug,
                         test_mode=dataset.test_mode, num_views=dataset.num_views)

        self.indices = indices

    def __getitem__(self, idx):
        idx = self.indices[idx]
        return super().__getitem__(idx)

    def __len__(self):
        return len(self.indices)


def create_folder(log_dir):
    # make summary folder
    new_dir="run/"+log_dir
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(new_dir)
        os.mkdir(new_dir)
    return new_dir

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

if __name__ == '__main__':
    args = parser.parse_args()

    pretraining = True
    # pretraining = True
    log_dir = args.name
    # log_dir.join(args.name)
    create_folder(args.name)
    config_f = open(os.path.join("run/"+log_dir, 'config.json'), 'w') 
    
    json.dump(vars(args), config_f)
    config_f.close()

    # STAGE 1
    log_dir = args.name+'_stage_1'
    create_folder(log_dir)#创建保存模型的文件夹
    # cnet = SVCNN(args.name, gender_classes=2, ethnic_classes=2, pretraining=pretraining, cnn_name=args.cnn_name) #创建单视角训练模型
    # optimizer = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)# 优化器，需要优化的参数，学习率，学习率衰减系数

    n_models_train = args.num_models*args.num_views #模型训练数
    
    #print(args.train_path)
    train_dataset = SingleImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
    # print(train_dataset)
    
    num_samples = len(train_dataset)
    # print(num_samples)
    # 交叉验证法
    num_folds = 5

    
    samples_per_fold = num_samples // num_folds
    # print(samples_per_fold)
    avg_sex_acc = 0.0
    avg_ethnic_acc = 0.0
    
    for fold in range(num_folds):
        print(f"Training Fold {fold}...")
        
        start_idx = fold * samples_per_fold
        end_idx = (fold + 1) * samples_per_fold
        val_indices = list(range(start_idx, end_idx))
        # print("val_indices:")
        # print(val_indices)
        train_indces = list(set(range(num_samples)) - set(val_indices))
        
        dataset_val = CustomSubset1(train_dataset, val_indices)
        
        dataset_train = CustomSubset1(train_dataset, train_indces)
       
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=0)
        
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=0)
       

        cnet = SVCNN(args.name, gender_classes=2, ethnic_classes=2, pretraining=pretraining,
                     cnn_name=args.cnn_name)  
        cnet.apply(weights_init)
        optimizer = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)  

        
        fold_log_dir = os.path.join('run/' + args.name + '_fold_' + str(fold), log_dir)
        trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, 'svcnn', fold_log_dir, num_views=1)
        



    # STAGE 2
    log_dir = args.name+'_stage_2'
    create_folder(log_dir)
    
    
    train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
   
    num_samples = len(train_dataset)

    # 交叉验证法
    num_folds = 5

    
    samples_per_fold = num_samples // num_folds

    avg_sex_acc = 0.0
    avg_ethnic_acc = 0.0
    
    for fold in range(num_folds):
        print(f"Training Fold {fold}...")
        
        start_idx = fold * samples_per_fold
        end_idx = (fold + 1) * samples_per_fold
        val_indices = list(range(start_idx, end_idx))
        train_indces = list(set(range(num_samples)) - set(val_indices))
       
        dataset_train = CustomSubset2(train_dataset, train_indces)
        dataset_val = CustomSubset2(train_dataset, val_indices)

       
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchSize, shuffle=False, num_workers=0)
        
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batchSize, shuffle=False, num_workers=0)
        cnet_2 = MVCNN(args.name, cnet, gender_classes=2, ethnic_classes=2, cnn_name=args.cnn_name,
                       num_views=args.num_views)
        
        cnet_2.apply(weights_init)
        optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        
        fold_log_dir = os.path.join('run/' + args.name + '_fold_' + str(fold), log_dir)
        trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, 'mvcnn', fold_log_dir,
                                  num_views=args.num_views)
        
        training_time_multi_task = trainer.train_model_and_measure_time(100)
        print(training_time_multi_task)



