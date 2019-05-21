import torch
import torch.nn as nn
import numpy as np
from attentionConv2d import AttentionConv2d


class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, input_dim, output_dim, stride=1, downsample_shortcut=None, attention=False, expansion=4,
                 kappa=None, nu=None, num_heads=None, H=None, W=None):
        super(BottleneckBlock, self).__init__()
        self.expansion = expansion

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_dim)

        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_dim)

        expansion_dim = expansion * output_dim
        self.conv3 = None
        if attention:
            dk = round(kappa * expansion_dim)
            dv = round(nu * expansion_dim)
            h = comptue_dim(H, 1, 3, stride)
            w = comptue_dim(W, 1, 3, stride)

            self.conv3 = AttentionConv2d(input_dim, expansion_dim, dk, dv, num_heads,
                                         kernel_size=1,
                                         padding=0,
                                         height=int(h),
                                         width=int(w))

        else:
            self.conv3 = nn.Conv2d(output_dim, expansion_dim, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(expansion * output_dim)
        self.downsample_shortcut = downsample_shortcut

    def forward(self, x):
        print('x ', x.size())
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        print('out ', out.size())

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        print('out ', out.size())

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample_shortcut is not None:
            residual = self.downsample_shortcut(x)

        out += residual
        out = self.relu(out)

        return out

def comptue_dim(dim, padding, kernel_size, stride):
    print(dim)
    print(padding)
    print(kernel_size)
    print(stride)
    return np.floor((dim + 2*padding - kernel_size) / stride) + 1
