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
mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=True).cuda()
std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=True).cuda()

from CAT import CATLayer



