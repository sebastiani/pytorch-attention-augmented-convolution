import torch
import torch.nn as nn
import numpy as np
from attentionConv2d import AttentionConv2d


class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, input_dim, output_dim, stride=1, downsample_shortcut=None, attention=False, expansion=4,
                 kappa=None, nu=None, num_heads=None, H=None, W=None, rel_encoding=False):
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
            while dk % num_heads != 0:
                dk += 1

            dv = round(nu * expansion_dim)
            while dv % num_heads != 0:
                dv += 1

            h = int(comptue_dim(H, 1, 3, stride)) if H is not None else None
            w = int(comptue_dim(W, 1, 3, stride)) if W is not None else None

            self.conv3 = AttentionConv2d(input_dim, expansion_dim, dk, dv, num_heads,
                                         kernel_size=1,
                                         padding=0,
                                         height=h,
                                         width=w,
                                         rel_encoding=False)

        else:
            self.conv3 = nn.Conv2d(output_dim, expansion_dim, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(expansion * output_dim)
        self.downsample_shortcut = downsample_shortcut

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample_shortcut is not None:
            residual = self.downsample_shortcut(x)

        out += residual
        out = self.relu(out)

        return out

def comptue_dim(dim, padding, kernel_size, stride):
    return np.floor((dim + 2*padding - kernel_size) / stride) + 1
