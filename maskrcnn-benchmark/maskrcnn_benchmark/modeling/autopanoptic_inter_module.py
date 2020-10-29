import torch
from torch import nn
import numpy as np
from maskrcnn_benchmark.layers import Conv2d
import torch.distributed as dist
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.modeling.make_layers import group_norm
import torch.nn.functional as F
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.search_space import make_inter_layer, inter_ss_keys

@registry.INTER_MODULE.register('AutoPanoptic_Inter_module_search')
@registry.INTER_MODULE.register('AutoPanoptic_Inter_module')
class AutoPanoptic_Inter_module(nn.Module):
    def __init__(self, cfg, inchannel, outchannel, outlayer, architecture=None):
        super(AutoPanoptic_Inter_module, self).__init__()
        self.cfg = cfg.clone()
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.outlayer = outlayer
        if architecture is not None:
            assert len(architecture) == outlayer, "outlayer:{}, architecture:{}".format(outlayer, len(architecture))
        self.architecture = architecture

        self.inter_module = []
        for i in range(outlayer):
            inter_block = nn.ModuleList()
            if architecture is None:
                for j in range(len(inter_ss_keys)):
                    inter_block.append(make_inter_layer(cfg, inter_ss_keys[j], inchannel, outchannel))
            else:
                inter_block = make_inter_layer(cfg, inter_ss_keys[architecture[i]], inchannel, outchannel)
            self.inter_module.append(inter_block)
            self.add_module('inter_layer{}'.format(i), inter_block)

    def forward(self, x, rngs=None):
        if rngs is not None:
            assert len(rngs) == self.outlayer, 'rngs:{}, self.out_layer:{}'.format(rngs, self.outlayer)

        atten = []
        if rngs is not None:
            for i in range(len(rngs)):
                atten.append(self.inter_module[i][rngs[i]](x))
        else:
            for i in range(self.outlayer):
                atten.append(self.inter_module[i](x))

        return atten

def build_inter_module(cfg, inchannel, outchannel, outlayer, architecture=None):
    func = registry.INTER_MODULE[
        cfg.MODEL.INTER_MODULE.NAME
    ]
    return func(cfg, inchannel, outchannel, outlayer, architecture=architecture)

if __name__ == '__main__':
    from maskrcnn_benchmark.config import cfg
    inter_module = build_inter_module(cfg, 256, 256, 4)
    print(inter_module)
