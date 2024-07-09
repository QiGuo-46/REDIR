import torch
import torch.nn as nn
import numpy as np

#from Networks.networks import ResnetBlock, get_norm_layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RegistrationNet(nn.Module):
    def __init__(self):
        super(RegistrationNet, self).__init__()

        def conv2Layer(inDim, outDim, ks, s, p):
            conv = nn.Conv2d(inDim, outDim, kernel_size=ks, stride=s, padding=p)
            norm = nn.InstanceNorm2d(outDim, affine=False, track_running_stats=False)
            relu = nn.ReLU(True)
            seq = nn.Sequential(*[conv, norm, relu])
            return seq

        self.convBlock1 = conv2Layer(60, 64, 7, 1, 0)
        self.convBlock2 = conv2Layer(64, 128, 3, 2, 0)
        self.convBlock3 = conv2Layer(128,256,3,2,0)
        #self.maxpooling=nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpooling = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(31*42*256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2*30)
        self.relu = nn.ReLU(True)

    def forward(self, inp):
        inp = inp.to(device)
        b, s, c, h, w = inp.shape
        x = inp.view(b, s * c, h, w)

        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        #x = torch.max(x, 2, keepdim=True)[0]
        #x = x.view(x.size(0), -1) #reshape
        x = self.maxpooling(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        x = x.view(-1,2,30)

        return x


