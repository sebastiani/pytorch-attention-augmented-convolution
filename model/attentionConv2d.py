import torch
import torch.nn as nn
from torch import einsum


class AttentionConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, dk, dv, num_heads, kernel_size):
        super(AttentionConv2d, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dk = dk
        self.dv = dv
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.dkh = self.dk // self.num_heads

        self.conv_qkv = nn.Conv2d(input_dim, 2*dk + dv, 1)
        self.conv_attn = nn.Conv2d(dv, dv, 1)
        self.conv_out = nn.Conv2d(input_dim, output_dim - dv, kernel_size)
        self.softmax = nn.Softmax()

    def forward(self, input):
        conv_out = self.conv_out(input)

        qkv = self.conv_qkv(input)    # batch_size, 2*dk+dv, H, W
        q, k, v = torch.split(qkv, [self.dk, self.dk, self.dv], dim=1)
        batch_size, _, H, W = q.size()
        q = q.view([batch_size, self.num_heads, self.dk // self.num_heads, H*W])
        k = k.view([batch_size, self.num_heads, self.dk // self.num_heads, H*W])
        v = v.view([batch_size, self.num_heads, self.dv // self.num_heads, H*W])

        q *= self.dkh ** -0.5
        weights = self.softmax(einsum('ijkl, ijkm -> ijlm', q, k))
        attn_out = einsum('ijkl, ijfl -> ijfk', weights, v)
        attn_out = attn_out.view(batch_size, self.dv, H, W)
        attn_out = self.conv_attn(attn_out)
        return torch.cat([conv_out, attn_out], dim=1)