import torch
from torch import nn
import numpy as np
from maskrcnn_benchmark.layers import Conv2d
# from maskrcnn_benchmark.layers import GroupNorm
from maskrcnn_benchmark.layers.misc import DFConv2d
import torch.distributed as dist
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.modeling.make_layers import GroupNorm
import torch.nn.functional as F
import torch.distributed as dist


head_ss_keys = [
    "DWConv3x3", 
    "DWConv5x5", 
    "DFConv3x3", 
    "DFConv5x5", 
    "ATConv3x3", 
    "ATConv5x5",
]

inter_ss_keys = [
    "Inter-r4",
    "Inter-r8",
    "Inter-r16",
]


class dwconv(nn.Module):
    def __init__(self, kernel_size, inchannel, outchannel, stride=1):
        super(dwconv, self).__init__()
        self.depthwise = Conv2d(inchannel, inchannel, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=inchannel)
        self.pointwise = Conv2d(inchannel, outchannel, kernel_size=1, stride=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.depthwise.weight)
        nn.init.constant_(self.depthwise.bias, 0)
        nn.init.xavier_normal_(self.pointwise.weight)
        nn.init.constant_(self.pointwise.bias, 0)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

    def __repr__(self):
        return "DWConv:{{\n\tdepthwise:[kernel_size:{}, inchannel:{}, outchannel:{}, stride:{}, padding:{}, groups:{}],\n\tpointwise:[inchannel:{}, outchannel:{}]\n}}".format(self.depthwise.kernel_size, self.depthwise.in_channels, self.depthwise.out_channels, self.depthwise.stride, self.depthwise.padding, self.depthwise.groups,
                           self.pointwise.in_channels, self.pointwise.out_channels)

class atconv(nn.Module):
    def __init__(self, kernel_size, inchannel, outchannel, stride=1, dilation=2):
        super(atconv, self).__init__()
        self.conv = Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=kernel_size-1, dilation=dilation) # consider dilation in padding
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        out = self.conv(x)
        return out

    def __repr__(self):
        return "ATConv:{{kernel_size:{}, inchannel:{}, outchannel:{}, dilation:{}, stride:{}, padding:{}}}".format(self.conv.kernel_size, self.conv.in_channels, self.conv.out_channels, self.conv.dilation, self.conv.stride, self.conv.padding)


class dfconv(nn.Module):
    def __init__(self, kernel_size, inchannel, outchannel, stride=1, dilation=1):
        super(dfconv, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = inchannel
        self.out_channels = outchannel
        self.stride = stride
        self.dialtion = dilation
        self.padding = (kernel_size - 1) // 2
        self.conv = DFConv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=self.padding)
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x):
        out = self.conv(x)
        return out

    def __repr__(self):
        return "DFConv:{{kernel_size:{}, inchannel:{}, outchannel:{}, stride:{}, padding:{}}}".format(self.kernel_size, self.in_channels, self.out_channels, self.stride, self.padding)

def make_layer(layer_name, inchannel, outchannel, relu=False, gn=False):
    layer = []
    assert layer_name in head_ss_keys

    kernel_size = int(layer_name.split('x')[1][0])
    if 'DW' in layer_name:
        layer.append(dwconv(kernel_size=kernel_size, inchannel=inchannel, outchannel=outchannel))
    elif 'AT' in layer_name:
        layer.append(atconv(kernel_size=kernel_size, inchannel=inchannel, outchannel=outchannel))
    elif 'DF' in layer_name:
        layer.append(dfconv(kernel_size=kernel_size, inchannel=inchannel, outchannel=outchannel))
    else:
        raise NotImplementedError("layer {} is not in search space".format(layer_name))

    if gn:
        layer.append(GroupNorm(outchannel, groups=32))
    if relu:
        layer.append(nn.ReLU(True))
    
    return nn.Sequential(*layer)


class SELayer(nn.Module):
    def __init__(self, cfg, inchannel, outchannel, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.inchannel = inchannel
        self.outchannel = outchannel       
        fc_dim = outchannel // reduction
        self.fc = nn.Sequential(
            nn.Linear(inchannel, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, outchannel),
            nn.Sigmoid()
        )

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        b, c, _, _ = x.size()
        # if b == 0: # no masks
        #     return torch.zeros(len(split), self.outchannel, 1, 1).cuda()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).unsqueeze(-1).unsqueeze(-1)
        y = torch.ones(y.size()).cuda() - y # negative attention
        return y # return attention

def make_inter_layer(cfg, layer_name, inchannel, outchannel):
    assert layer_name in inter_ss_keys
    ratio = int(layer_name[-1])
    return SELayer(cfg, inchannel, outchannel, reduction=ratio)
    