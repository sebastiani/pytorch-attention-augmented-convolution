import torch
import torch.nn as nn
from .attentionConv2d import AttentionConv2d
from ..utils.utils import comptue_dim


class BottleneckBlock(nn.Module):
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
            self.conv3 = AttentionConv2d(output_dim, expansion_dim, dk, dv, num_heads, kernel_size=1, padding=0,
                                         height=h, width=w)

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
