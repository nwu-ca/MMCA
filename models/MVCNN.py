import numpy as np
import os
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as model
from .base_densenet import densenet121
from .base_googlenet import googlenet
from .base_resnet import resnet18, resnet34, resnet50
from .od_resnet import od_resnet18
from .base_vgg import vgg11, vgg13, vgg16
from .base_alexnet import alexnet
from .ParC_resnet50 import parc_res50
from .base_mobilenet import mobilenet_v2
from .Model import Model
from repvgg import create_RepVGG_A0
from .ConvNext import convnext_tiny
# from torchvision.models import MobileNetV2
# from .mobileVit import mobile_vit_xx_small
mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=True).cuda()
std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=True).cuda()

from CAT import CATLayer

def flip(x, dim):
    xsize = x.size()
    
    dim = x.dim() + dim if dim < 0 else dim
    
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class SVCNN(Model):

    def __init__(self, name, gender_classes=2, ethnic_classes=2, pretraining=True, cnn_name='vgg11'):
        super(SVCNN, self).__init__(name)

        self.ethnic_classname = ['han', 'wei']
        self.gender_classname = ['man', 'woman']
        self.ethnic_classes = ethnic_classes
        self.gender_classes = gender_classes
        
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        
        self.use_resnet = cnn_name.startswith('resnet')
        # self.use_mobilenet = cnn_name.startswith('mobile')
        self.use_googlenet = cnn_name.startswith('google')
        self.use_repvgg = cnn_name.startswith('repvgg')
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=True).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=True).cuda()

        if self.use_resnet or self.use_googlenet or self.use_repvgg:
            if self.cnn_name == 'resnet18':
                model = resnet18(pretrained=self.pretraining)
                self.net = nn.Sequential(*list(model.children())[:-1])
                self.ethnic = nn.Sequential(nn.Linear(512, self.ethnic_classes))
                
                self.gender = nn.Sequential(nn.Linear(512, self.gender_classes))



            elif self.cnn_name == 'resnet34':
                model = resnet34(pretrained=self.pretraining)
                self.net = nn.Sequential(*list(model.children())[:-1])
                self.gender = nn.Sequential(nn.Linear(512, self.gender_classes))
                self.ethnic = nn.Sequential(nn.Linear(512, self.ethnic_classes))

            elif self.cnn_name == 'resnet50':
                model = resnet50(pretrained=self.pretraining)
                self.net = nn.Sequential(*list(model.children())[:-1])
                self.gender = nn.Sequential(nn.Linear(2048, self.gender_classes))
                self.ethnic = nn.Sequential(nn.Linear(2048, self.ethnic_classes))

            elif self.cnn_name=='resnet18_od':
                self.net = od_resnet18(pretraining=self.pretraining)
                self.net.gender = nn.Linear(512, self.gender_classes)
                self.net.ethnic = nn.Linear(512, self.ethnic_classes)
            elif self.cnn_name == 'mobile_vit_xx_small':
               
                
            elif self.cnn_name=='resnet50_parc':
                self.net=parc_res50(pretrained=self.pretraining)
                
                self.net.fc = nn.Linear(512, 4)
            elif self.cnn_name == 'googlenet':
                model = googlenet(pretrained=self.pretraining)
                self.net = nn.Sequential(*list(model.children())[:-1])
                
                self.gender = nn.Sequential(nn.Linear(1024, self.gender_classes))
                
                self.ethnic = nn.Sequential(nn.Linear(1024, self.ethnic_classes))
                
            elif self.cnn_name == 'repvggA0':
                model = create_RepVGG_A0()
                self.net = nn.Sequential(*list(model.children())[:-1])
                self.gender = nn.Sequential(nn.Linear(512, self.gender_classes))
                self.ethnic = nn.Sequential(nn.Linear(512, self.ethnic_classes))

        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = alexnet(pretrained=self.pretraining).features
                self.gender = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(256 * 6 * 6, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, self.gender_classes),
                )
                self.ethnic = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(256 * 6 * 6, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, self.ethnic_classes),
                )

            elif self.cnn_name == 'vgg11':
                self.net_1 = vgg11(pretrained=self.pretraining).features
                self.gender = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(0.25),
                    nn.Linear(4096, self.gender_classes),
                )
                self.ethnic = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(0.5),
                    nn.Linear(4096, self.ethnic_classes),
                )
            elif self.cnn_name == 'vgg16':
                self.net_1 = vgg16(pretrained=self.pretraining).features
                self.gender = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(0.5),
                    nn.Linear(4096, self.gender_classes),
                )
                self.ethnic = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(0.5),
                    nn.Linear(4096, self.ethnic_classes),
                )
            elif self.cnn_name == 'mobilenet':
                model = mobilenet_v2(pretrained=self.pretraining)
                
                self.net_1 = nn.Sequential(*list(model.children())[:-1])

               
                self.gender = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(1280, self.gender_classes),
                )
               
                self.ethnic = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(1280, self.ethnic_classes),
                )
            elif self.cnn_name == 'densenet':
                self.net_1 = densenet121(pretrained=self.pretraining).features
                
                self.gender = nn.Linear(1024*7*7, self.gender_classes)
                self.ethnic = nn.Linear(1024*7*7, self.ethnic_classes)
            elif self.cnn_name == 'convnext':
                model = convnext_tiny(pretrained=self.pretraining, in_22k=False)
                
                self.net_1 = model
                self.gender = nn.Sequential(nn.Linear(768, self.gender_classes))
                self.ethnic = nn.Sequential(nn.Linear(768, self.ethnic_classes))
            
    def forward(self, x):
        if self.use_resnet or self.use_mobilenet or self.use_googlenet:
            x = self.net(x)
            x = torch.flatten(x, 1)
            return {
            'gender': self.gender(x),
            'ethnic': self.ethnic(x)
        }
        else:
            y = self.net_1(x)
            y = y.view(y.shape[0], -1)
            return {
                'gender': self.gender(y),
                'ethnic': self.ethnic(y)
            }


class MVCNN(Model):

    def __init__(self, name, model, gender_classes=2, ethnic_classes=2, cnn_name='vgg11', num_views=12):
        super(MVCNN, self).__init__(name)

        self.ethnic_classname = ['han', 'wei', ]
        self.gender_classname = ['man', 'woman']
        self.ethnic_classes = ethnic_classes
        self.gender_classes = gender_classes

        self.num_views = num_views
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=True).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=True).cuda()

        self.use_resnet = cnn_name.startswith('resnet')
        self.use_repvgg = cnn_name.startswith('repvgg')
        self.use_googlenet = cnn_name.startswith('google')
        if self.use_resnet or self.use_googlenet or self.use_repvgg:
            
            self.net_1 = model.net
            if cnn_name == 'resnet18':
                self.gender = nn.Sequential(nn.Linear(512, self.gender_classes))
                
                self.ethnic = nn.Sequential(nn.Linear(512, self.ethnic_classes))

            elif cnn_name == 'resnet34':
                self.gender = nn.Sequential(nn.Linear(512, self.gender_classes))
                self.ethnic = nn.Sequential(nn.Linear(512, self.ethnic_classes))

            elif cnn_name == 'resnet50':
                self.gender = nn.Sequential(nn.Linear(2048, self.gender_classes))
                self.ethnic = nn.Sequential(nn.Linear(2048, self.ethnic_classes))

            elif cnn_name == 'googlenet':
                self.gender = nn.Sequential(nn.Linear(1024, self.gender_classes))
                self.ethnic = nn.Sequential(nn.Linear(1024, self.ethnic_classes))
            elif cnn_name == 'repvggA0':
                self.gender = nn.Sequential(nn.Linear(512, self.gender_classes))
                self.ethnic = nn.Sequential(nn.Linear(512, self.ethnic_classes))
        

        else:
            self.net_1 = model.net_1
            
            if cnn_name == 'alexnet':
                self.gender = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(256 * 6 * 6, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, self.gender_classes),
                )
                self.ethnic = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(256 * 6 * 6, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, self.ethnic_classes),
                )
            elif cnn_name == 'vgg11':
                self.gender = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(0.5),
                    nn.Linear(4096, self.gender_classes),
                )
                self.ethnic = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(0.5),
                    nn.Linear(4096, self.ethnic_classes),
                )
            elif cnn_name == 'vgg16':
                self.gender = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(0.5),
                    nn.Linear(4096, self.gender_classes),
                )
                self.ethnic = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(0.5),
                    nn.Linear(4096, self.ethnic_classes),
                )
            elif cnn_name == 'mobilenet':
                self.gender = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(1280, self.gender_classes),
                )
                self.ethnic = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(1280, self.ethnic_classes),
                )
            elif cnn_name == 'densenet':

                self.gender = nn.Sequential(nn.Linear(1024*7*7, self.gender_classes))
                self.ethnic = nn.Sequential(nn.Linear(1024*7*7, self.ethnic_classes))

            elif cnn_name == 'convnext':

                self.gender = nn.Sequential(nn.Linear(768*7*7, self.gender_classes))
                self.ethnic = nn.Sequential(nn.Linear(768*7*7, self.ethnic_classes))

        
        # self.CAT = CATLayer(dim=512, H=1, W=1*12, num_heads=2, patch_size=1, mlp_ratio=4., qkv_bias=True,
        #          qk_scale=None, drop=0., ipsa_attn_drop=0.25, cpsa_attn_drop=0., drop_path=0.,
        #          norm_layer=nn.LayerNorm, use_checkpoint=False)

    def forward(self, x):
       
        y = self.net_1(x) 
        H = y.shape[-2]
        W = y.shape[-1]

        
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))
        
    
        y = torch.max(y, 1)[0].view(y.shape[0], -1)
        
        return {
            'gender': self.gender(y),
            'ethnic': self.ethnic(y)
        }

