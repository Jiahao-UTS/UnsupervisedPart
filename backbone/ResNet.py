from collections import OrderedDict
from typing import Optional, List
import copy
from typing import Tuple, Union

import numpy as np
import math
import torch
import os
import torch.nn.functional as F
from torch import nn
from timm.models.layers import trunc_normal_, DropPath
from timm.models.vision_transformer import VisionTransformer
from timm.models.resnet import Bottleneck, BasicBlock, ResNet
from functools import partial

import logging

logger = logging.getLogger(__name__)

class CNN_Encoder(ResNet):
    def __init__(self, **kwargs):
        super(CNN_Encoder, self).__init__(**kwargs)
        self.head = nn.Identity()
        self.layer2 = nn.Identity()
        self.layer3 = nn.Identity()
        self.layer4 = nn.Identity()
        self.apply(self.init_weights)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)

        return x1

    def forward(self, x):
        x = self.forward_features(x)
        return x


def Get_CNN_Encoder(model_name, pretrain_root=''):
    if model_name == 'ResNet50':
        model = CNN_Encoder(block=Bottleneck, layers=[3, 4, 6, 3])
        if os.path.isfile(os.path.join(pretrain_root, model_name.split('_')[0] + '.pth')):
            load_model(model, os.path.join(pretrain_root, model_name.split('_')[0] + '.pth'))
        return model
    elif model_name == 'ResNet34':
        model = CNN_Encoder(block=BasicBlock, layers=[3, 4, 6, 3])
        if os.path.isfile(os.path.join(pretrain_root, model_name.split('_')[0] + '.pth')):
            load_model(model, os.path.join(pretrain_root, model_name.split('_')[0] + '.pth'))
        return model
    elif model_name == 'ResNet18':
        model = CNN_Encoder(block=BasicBlock, layers=[2, 2, 2, 2])
        if os.path.isfile(os.path.join(pretrain_root, model_name.split('_')[0] + '.pth')):
            load_model(model, os.path.join(pretrain_root, model_name.split('_')[0] + '.pth'))
        return model
    else:
        raise NotImplementedError


def load_model(model, path):
    pretrained_dict = torch.load(path)
    # print('=> loading pretrained model {}'.format(path))
    logger.info('=> loading pretrained model {}'.format(path))
    model_dict = model.state_dict()
    # print(model_dict.keys())
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info('=> loading {} pretrained model {}'.format(k, path))
        # print('=> loading {} pretrained model {}'.format(k, path))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
