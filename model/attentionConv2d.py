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
        logits = einsum('ijkl, ijkm -> ijlm', q, k)
        if self.relative_encoding:
            h_rel_logits, w_rel_logits = self._relative_logits(q, H, W)
            logits += h_rel_logits
            logits += w_rel_logits

        weights = self.softmax(logits)
        attn_out = einsum('ijkl, ijfl -> ijfk', weights, v)
        attn_out = attn_out.view(batch_size, self.dv, H, W)
        attn_out = self.conv_attn(attn_out)
        return torch.cat([conv_out, attn_out], dim=1)


    def _relative_logits(self, q, H, W):
        b, nh, dkh, _ = q.size()
        q = q.view(b, nh, dkh, H, W)

        key_rel_w = dkh**-0.5 + torch.rand(2*W-1, dkh)
        key_rel_h = dkh**-0.5 + torch.rand(2*H-1, dkh)

        rel_logits_w = self._relative_logits1d(q, key_rel_w, H, W, nh, [0, 1, 2, 4, 3, 5])
        rel_logits_h = self._relative_logits1d(torch.transpose(q, [0, 1, 2, 4, 3]), key_rel_h, W, H, nh, [0, 1, 4, 2, 5, 3])
        return rel_logits_h, rel_logits_w

    def _relative_logits1d(self, q, rel_k, H, W, Nh, transpose_mask):
        rel_logits = einsum('bhdxy, md -> bhxym', q, rel_k)

        rel_logits = rel_logits.view([-1, Nh*H, W, 2*W-1])
        rel_logits = self._rel_to_abs(rel_logits)
        rel_logits = rel_logits.view([-1, Nh, H, W, W]).unsqueeze(dim=3).repeat([1,1,1,H,1,1])
        rel_logits = rel_logits.transpose(transpose_mask)
        rel_logits = rel_logits.view([-1, Nh, H*W, H*W])
        return rel_logits

    def _rel_to_abs(self, x):
        b, nh, l, _ = x.size()

        col_pad = torch.zeros((b, nh, l, 1))
        x = torch.cat([x, col_pad], dim=3)
        flat_x = x.view([b, nh, l*(2*l)])
        flat_pad = torch.zeros((b, nh, l-1))
        flat_x_padded = torch.cat([flat_x, flat_pad], dim=2)

        final_x = flat_x_padded.view([b, nh, l+1, 2*l-1])
        final_x = final_x[:, :, :l, l-1:]
        return final_x
