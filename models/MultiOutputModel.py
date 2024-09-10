import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .base_resnet import resnet18,resnet34,resnet50

class MultiOutputModel(nn.Module):
    def __init__(self, gender_classes, ethnic_classes):
        super().__init__()
        self.base_model = resnet18()
        self.base_model = models.mobilenet_v2().features  # take the model without classifier
        last_channel = models.mobilenet_v2().last_channel  # size of the layer before classifier


        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create separate classifiers for our outputs
        self.gender = nn.Linear(512, gender_classes)
        self.ethnic = nn.Linear(512, ethnic_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)

        return {
            'gender': self.gender(x),
            'ethnic': self.ethnic(x)
        }

    # def get_loss(self, net_output, ground_truth):
    #     color_loss = F.cross_entropy(net_output['color'], ground_truth['color_labels'])
    #     gender_loss = F.cross_entropy(net_output['gender'], ground_truth['gender_labels'])
    #     article_loss = F.cross_entropy(net_output['article'], ground_truth['article_labels'])
    #     loss = color_loss + gender_loss + article_loss
    #     return loss, {'color': color_loss, 'gender': gender_loss, 'article': article_loss}
