import os
import torch
import warnings
import torch.nn as nn
import sys
import math

sys.path.append("..")
import numpy as np
import torch.nn.functional as F
from utils import get_feature_module, vgg16_loss
from torch.autograd import Variable


warnings.filterwarnings("ignore")


class EventWarping(torch.nn.Module):
    def __init__(self, config, device):
        self.device = device

    def smooth(self, outputs):
        # conventional total variation with forward differences
        c = outputs.shape[1]
        w = outputs.shape[2]
        h = outputs.shape[3]
        img_dx = torch.abs(outputs[:, :, :-1, :] - outputs[:, :, 1:, :])
        img_dy = torch.abs(outputs[:, :, :, :-1] - outputs[:, :, :, 1:])
        tv_error = (img_dx.sum()**2 + img_dy.sum()**2)**0.5

        return tv_error / (c * w * h)
        return tv_error / (c * w * h)

    def pix(self, outputs, occ_free_aps):
        L1 = nn.L1Loss(reduction="mean")
        return L1(outputs, occ_free_aps)

    def per(self, outputs, occ_free_aps):
        loss = 0
        layer_indexs = [3, 8, 15, 22]
        Lambdas = [0.1, 1 / 21, 10 / 21, 10 / 21]
        outputs = torch.cat([outputs, outputs, outputs], dim=1)
        occ_free_aps = torch.cat([occ_free_aps, occ_free_aps, occ_free_aps], dim=1)
        for i, index in enumerate(layer_indexs):
            Lambda = Lambdas[i]
            feature_module = get_feature_module(index, self.device)
            loss += Lambda * vgg16_loss(
                feature_module, nn.MSELoss(), outputs, occ_free_aps
            )
        return loss / len(layer_indexs)


# 计算特征提取模块的感知损失
