import torch
import torch.nn as nn
import torch.nn.functional as F
from Networks.networks import ResnetBlock, get_norm_layer
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from Networks.attention import TCA, TCSA
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义Spatial Transformer Network（STN）模块
# 定义Spatial Transformer Network（STN）模块
class SpatialTransformerUNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SpatialTransformerUNet, self).__init__()
        # 下采样层
        self.downsampling1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.downsampling2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.downsampling3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        # STN局部化网络
        self.localization1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.localization2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.localization3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # 回归头部用于输出（4，2，3）的变换矩阵
        self.fc_loc1 = nn.Sequential(
            nn.Linear(128 * 128, 512),
            nn.Linear(512, 2 * 3),  # 输出（4，2，3）的变换矩阵
            nn.Tanh()
        )
        self.fc_loc2 = nn.Sequential(
            nn.Linear(64 * 64, 512),
            nn.Linear(512,  2 * 3),
            nn.Tanh()
        )
        self.fc_loc3 = nn.Sequential(
            nn.Linear( 32 * 32, 512),
            nn.Linear(512,  2 * 3),
            nn.Tanh()
        )

        # 上采样层
        self.upsampling1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upsampling2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upsampling3 = nn.Sequential(
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # 输出图像像素值范围 [0, 1]
        )
        self.stn = TCSA(64,64)

    def forward(self, x):
        # 获取STN的变换参数
        batch_size, time,channels, height, width = x.size()
        x = x.view(batch_size , time*channels, height, width)  # 合并批次和时间维度

        # 第一层下采样和STN
        xd1 = self.downsampling1(x)
        xd1 = self.stn(x)
        xs1 = self.localization1(xd1)
        xs1 = xs1.view(batch_size, 32, -1)
        #xs1 = xs1.view(batch_size * 32, -1)

        theta1 = self.fc_loc1(xs1)
        theta1 = theta1.view(batch_size,32, 2, 3)  # 变换参数为（4，2，3）
        xd1 = xd1.view(batch_size,32,2,128,128)

        x1 = torch.zeros_like(xd1).to(xd1.device)
        for i in range(batch_size):  # batch
            grid = F.affine_grid(theta1[i,:], xd1[i,:].squeeze().size())
            x1[i] = F.grid_sample(xd1[i,:], grid)

        x1 = x1.view(batch_size, 64, 128, 128)  # 合并批次和时间维度

        # 第二层下采样和STN
        xd2 = self.downsampling2(x1)
        xs2 = self.localization2(xd2)
        xs2 = xs2.view(batch_size,64, -1)
        #xs2 = xs2.view(batch_size * time, -1)

        theta2 = self.fc_loc2(xs2)
        theta2 = theta2.view(batch_size,64, 2, 3)
        xd2 = xd2.view(batch_size, 64, 2, 64, 64)

        x2 = torch.zeros_like(xd2).to(xd2.device)
        for i in range(batch_size):  # batch
            grid = F.affine_grid(theta2[i, :], xd2[i, :].squeeze().size())
            x2[i] = F.grid_sample(xd2[i, :], grid)

        x2 = x2.view(batch_size , 128, 64, 64)  # 合并批次和时间维度

        # 第三层下采样和STN
        xd3 = self.downsampling3(x2)
        xs3 = self.localization3(xd3)
        xs3 = xs3.view(batch_size, 128, -1)
        #xs3 = xs3.view(batch_size * time, -1)

        theta3 = self.fc_loc3(xs3)
        theta3 = theta3.view(batch_size, 128, 2, 3)
        xd3 = xd3.view(batch_size, 128, 2, 32, 32)

        x3 = torch.zeros_like(xd3).to(xd3.device)
        for i in range(batch_size):  # batch
            grid = F.affine_grid(theta3[i,  :], xd3[i,  :].squeeze().size())
            x3[i] = F.grid_sample(xd3[i, :], grid)

        #x3 = x3.view(batch_size * time, 128, 64, 64)  # 合并批次和时间维度

        # 上采样
        x3 = x3.view(batch_size, 256, 32, 32)  # 将通道重新调整为256，尺寸减小到16x16
        x3 = self.upsampling1(x3)
        x3 = self.upsampling2(x3)
        x3 = self.upsampling3(x3)

        x3 = x3.view(batch_size, time, channels, height, width)  # 还原到原始尺寸
        return x3

class SpatialTransformerUNet1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SpatialTransformerUNet1, self).__init__()
        # 下采样层
        # STN局部化网络

        self.localization1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.downsampling1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.downsampling2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.downsampling3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.localization3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # 回归头部用于输出（4，2，3）的变换矩阵
        self.fc_loc1 = nn.Sequential(
            nn.Linear( 2*256 * 256, 512),
            nn.Linear(512, 2 * 3),  # 输出（4，2，3）的变换矩阵
            nn.Tanh()
        )
        self.fc_loc3 = nn.Sequential(
            nn.Linear( 32 * 32, 512),
            nn.Linear(512,  2 * 3),
            nn.Tanh()
        )
        #self.resBlock1 = ResnetBlock(256, padding_type='reflect', norm_layer=get_norm_layer('instance'), use_dropout=True, use_bias=True)
        #self.resBlock2 = ResnetBlock(256, padding_type='reflect', norm_layer=get_norm_layer('instance'), use_dropout=True, use_bias=True)
       # self.resBlock3 = ResnetBlock(256, padding_type='reflect', norm_layer=get_norm_layer('instance'), use_dropout=True, use_bias=True)
        # 上采样层
        self.upsampling1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upsampling2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upsampling3 = nn.Sequential(
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # 输出图像像素值范围 [0, 1]
        )
        self.stn = TCSA(64, 64)
        self.resBlock1 = ResnetBlock(256, padding_type='reflect', norm_layer=get_norm_layer('instance'), use_dropout=True, use_bias=True)


    def forward(self, x):
        print("===============================")
        print(x.size())
        # 获取STN的变换参数
        batch_size, time,channels, height, width = x.size()
        #x = x.view(batch_size * time, channels, height, width)  # 合并批次和时间维度

        xs1 = x.view(batch_size ,time*channels, height, width)

        xs1 = self.localization1(xs1)
        xs1 = xs1.view(batch_size, time, -1)
        theta1 = self.fc_loc1(xs1)
        theta1 = theta1.view(batch_size,time, 2, 3)  # 变换参数为（4，2，3）

        t=torch.Tensor([[1,0,0],[0,1,0]])
        t=t.expand(batch_size,time,2, 3)
        t = t.to(theta1.device)
        theta1=t+theta1*0.05



        x1 = torch.zeros_like(x).to(x.device)
        for i in range(batch_size):  # batch
            grid = F.affine_grid(theta1[i, :], x[i, :].squeeze().size())
            x1[i,:] = F.grid_sample(x[i, :], grid)

        x1 = x1.view(batch_size ,time*2, 256, 256)  # 合并批次和时间维度

        # 第二层下采样和STN
        xd1 = self.downsampling1(x1)
        # att = self.stn(xd1)
        xd2 = self.downsampling2(xd1)
        xd3 = self.downsampling3(xd2)

        xs3 = self.localization3(xd3)
        xs3 = xs3.view(batch_size, 128, -1)
        theta3 = self.fc_loc3(xs3)
        theta3 = theta3.view(batch_size, 128, 2, 3)

        t3 = torch.Tensor([[1, 0, 0], [0, 1, 0]])
        t3 = t3.expand(batch_size, 128, 2, 3)
        t3 = t3.to(theta3.device)
        theta3 = t3 + theta3 * 0.05

        xd3 = xd3.view(batch_size,  128, 2, 32, 32)
        x3 = torch.zeros_like(xd3).to(xd3.device)
        for i in range(batch_size):  # batch
            grid = F.affine_grid(theta3[i, :], xd3[i, :].squeeze().size())
            x3[i] = F.grid_sample(xd3[i, :], grid)

        #x3 = x3.view(batch_size * time, 128, 64, 64)  # 合并批次和时间维度

        # 上采样
        x3 = x3.view(batch_size, 256, 32, 32)  # 将通道重新调整为256，尺寸减小到16x16

        #x3 = self.resBlock1(x3)
        #x3 = self.resBlock2(x3)
        #x3 = self.resBlock3(x3)


        x3 = self.upsampling1(x3)
        x3 = self.upsampling2(x3)
        x3 = self.upsampling3(x3)

        x3 = x3.view(batch_size, time, channels, height, width)  # 还原到原始尺寸
        return x3

