"""ECB — Edge-oriented Convolution Block (Zhang et al., ECBSR, ACM MM '21).

Train-time: four parallel branches summed together
  1. plain 3x3
  2. expand-and-squeeze (1x1 -> D=2*out -> 3x3)
  3. scaled Sobel-x   (1x1 -> depthwise fixed Sobel, learned per-channel scale)
  4. scaled Sobel-y
  5. scaled Laplacian (2nd-order)
Inference (eval mode): the branches fold into ONE 3x3 conv, so a deployed ECB
costs exactly the same as nn.Conv2d(in, out, 3, padding=1). It is a drop-in for
that call -- activation stays outside, in the surrounding block.

The fold is exact (train-forward == eval-forward to fp tolerance); run this file
to check. Reference: https://github.com/xindongzhang/ECBSR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqConv3x3(nn.Module):
    # one ECB branch: a 1x1 conv (k0,b0) followed by either a 3x3 conv or a
    # depthwise fixed edge filter. rep_params() folds the pair into one 3x3.
    def __init__(self, seq_type, in_planes, out_planes, depth_multiplier=2.0):
        super().__init__()
        self.type = seq_type
        self.in_planes = in_planes
        self.out_planes = out_planes

        if seq_type == "conv1x1-conv3x3":
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = nn.Conv2d(in_planes, self.mid_planes, 1)
            self.k0, self.b0 = conv0.weight, conv0.bias
            conv1 = nn.Conv2d(self.mid_planes, out_planes, 3)
            self.k1, self.b1 = conv1.weight, conv1.bias
            return

        # edge-filter branches: 1x1 expand, then a fixed Sobel/Laplacian stencil
        # scaled per channel. Only the 1x1 weights, the scale and the bias learn.
        conv0 = nn.Conv2d(in_planes, out_planes, 1)
        self.k0, self.b0 = conv0.weight, conv0.bias
        self.scale = nn.Parameter(torch.randn(out_planes, 1, 1, 1) * 1e-3)
        self.bias = nn.Parameter(torch.randn(out_planes) * 1e-3)

        mask = torch.zeros(out_planes, 1, 3, 3)
        if seq_type == "conv1x1-sobelx":
            for i in range(out_planes):
                mask[i, 0, 0, 0] = 1.0;  mask[i, 0, 1, 0] = 2.0;  mask[i, 0, 2, 0] = 1.0
                mask[i, 0, 0, 2] = -1.0; mask[i, 0, 1, 2] = -2.0; mask[i, 0, 2, 2] = -1.0
        elif seq_type == "conv1x1-sobely":
            for i in range(out_planes):
                mask[i, 0, 0, 0] = 1.0;  mask[i, 0, 0, 1] = 2.0;  mask[i, 0, 0, 2] = 1.0
                mask[i, 0, 2, 0] = -1.0; mask[i, 0, 2, 1] = -2.0; mask[i, 0, 2, 2] = -1.0
        elif seq_type == "conv1x1-laplacian":
            for i in range(out_planes):
                mask[i, 0, 0, 1] = 1.0; mask[i, 0, 1, 0] = 1.0
                mask[i, 0, 1, 2] = 1.0; mask[i, 0, 2, 1] = 1.0
                mask[i, 0, 1, 1] = -4.0
        else:
            raise ValueError(f"unknown SeqConv3x3 type: {seq_type}")
        self.register_buffer("mask", mask)

    def _pad_with_b0(self, y):
        # zero-pad, then overwrite the border with b0 so a single folded 3x3 on the
        # zero-padded input stays bit-equivalent: the 1x1 emits k0*x+b0 everywhere,
        # and outside the image x=0 -> the intermediate is b0, not 0.
        y = F.pad(y, (1, 1, 1, 1))
        b0 = self.b0.view(1, -1, 1, 1)
        y[:, :, 0:1, :] = b0
        y[:, :, -1:, :] = b0
        y[:, :, :, 0:1] = b0
        y[:, :, :, -1:] = b0
        return y

    def forward(self, x):
        if self.type == "conv1x1-conv3x3":
            y = F.conv2d(x, self.k0, self.b0)
            y = self._pad_with_b0(y)
            return F.conv2d(y, self.k1, self.b1)
        y = F.conv2d(x, self.k0, self.b0)
        y = self._pad_with_b0(y)
        return F.conv2d(y, self.scale * self.mask, self.bias, groups=self.out_planes)

    def rep_params(self):
        # fold (k0,b0) -> (k1,b1) into one 3x3 (RK, RB)
        if self.type == "conv1x1-conv3x3":
            k1, b1 = self.k1, self.b1
        else:
            stencil = self.scale * self.mask  # (out,1,3,3)
            k1 = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=stencil.device)
            for i in range(self.out_planes):
                k1[i, i] = stencil[i, 0]
            b1 = self.bias
        rk = F.conv2d(k1, self.k0.permute(1, 0, 2, 3))  # (out,in,3,3)
        rb = torch.ones(1, self.k0.size(0), 3, 3, device=k1.device) * self.b0.view(1, -1, 1, 1)
        rb = F.conv2d(rb, k1).view(-1) + b1
        return rk, rb


class ECB(nn.Module):
    """Drop-in for nn.Conv2d(in_planes, out_planes, 3, padding=1).

    Train mode runs the 4 branches; eval mode runs the single folded 3x3.
    """

    def __init__(self, in_planes, out_planes, depth_multiplier=2.0):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.conv3x3 = nn.Conv2d(in_planes, out_planes, 3, padding=1)
        self.es = SeqConv3x3("conv1x1-conv3x3", in_planes, out_planes, depth_multiplier)
        self.sbx = SeqConv3x3("conv1x1-sobelx", in_planes, out_planes)
        self.sby = SeqConv3x3("conv1x1-sobely", in_planes, out_planes)
        self.lpl = SeqConv3x3("conv1x1-laplacian", in_planes, out_planes)

    def forward(self, x):
        if self.training:
            return self.conv3x3(x) + self.es(x) + self.sbx(x) + self.sby(x) + self.lpl(x)
        rk, rb = self.rep_params()
        return F.conv2d(x, rk, rb, padding=1)

    def rep_params(self):
        rk, rb = self.conv3x3.weight, self.conv3x3.bias
        for branch in (self.es, self.sbx, self.sby, self.lpl):
            k, b = branch.rep_params()
            rk = rk + k
            rb = rb + b
        return rk, rb

    def to_plain(self):
        # bake the fold into a standalone Conv2d for deploy / ONNX export, so the
        # branch arithmetic doesn't end up in the traced graph.
        conv = nn.Conv2d(self.in_planes, self.out_planes, 3, padding=1)
        rk, rb = self.rep_params()
        with torch.no_grad():
            conv.weight.copy_(rk)
            conv.bias.copy_(rb)
        return conv


def convert_ecb_to_plain(model):
    # in-place: replace every ECB submodule with its folded Conv2d. Call on a trained
    # model (after .eval()) before ONNX/TRT export so the branch math isn't traced.
    for name, child in model.named_children():
        if isinstance(child, ECB):
            setattr(model, name, child.to_plain())
        else:
            convert_ecb_to_plain(child)
    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    print("ECB train-forward vs folded-3x3 equivalence:")
    for cin, cout in [(3, 32), (32, 32), (32, 64), (128, 256)]:
        ecb = ECB(cin, cout)
        # nudge params off their tiny init so the test is non-trivial
        for p in ecb.parameters():
            if p.requires_grad:
                p.data.add_(torch.randn_like(p) * 0.05)
        x = torch.randn(2, cin, 24, 24)
        ecb.train()
        yt = ecb(x)
        ecb.eval()
        with torch.no_grad():
            ye = ecb(x)
        diff = (yt - ye).abs().max().item()
        params_branched = sum(p.numel() for p in ecb.parameters() if p.requires_grad)
        rk, _ = ecb.rep_params()
        params_folded = rk.numel() + cout
        print(f"  in={cin:3d} out={cout:3d}  max|train-eval|={diff:.2e}  "
              f"train_params={params_branched:6d} -> deploy_params={params_folded:6d}")
