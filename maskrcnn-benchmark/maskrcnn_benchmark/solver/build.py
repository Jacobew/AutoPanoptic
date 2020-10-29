# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR
from maskrcnn_benchmark.modeling.search_space import head_ss_keys, inter_ss_keys
from maskrcnn_benchmark.modeling.backbone.search_space import blocks_key
import math

def make_optimizer(cfg, model):
    params = []
    d = []
    backbone_ss_size = len(blocks_key)
    head_ss_size = len(head_ss_keys)
    inter_ss_size = len(inter_ss_keys)
    if 'search' in cfg.MODEL.BACKBONE.CONV_BODY:
        _lcm = backbone_ss_size*head_ss_size//math.gcd(backbone_ss_size, head_ss_size)
        lcm = inter_ss_size*_lcm//math.gcd(inter_ss_size, _lcm)
    else:
        lcm = inter_ss_size*head_ss_size//math.gcd(inter_ss_size, head_ss_size)
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        if not cfg.NAS.TRAIN_SINGLE_MODEL:
            if 'search' in cfg.MODEL.BACKBONE.CONV_BODY:
                if 'backbone' in key: 
                    if 'first_conv' not in key:
                        lr /= (lcm / backbone_ss_size)
                    else:
                        lr /= lcm
                elif 'AutoPanoptic_mask_fcn' in key:
                    lr /= (lcm / head_ss_size)
                elif 'subnet' in key:
                    lr /= (lcm / head_ss_size)
                elif 'inter_layer' in key:
                    lr /= (lcm / inter_ss_size)
                else:
                    lr /= lcm
            else:
                if 'AutoPanoptic_mask_fcn' in key:
                    lr /= (lcm / head_ss_size)
                elif 'subnet' in key:
                    lr /= (lcm / head_ss_size)
                elif 'inter_layer' in key:
                    lr /= (lcm / inter_ss_size)
                else:
                    lr /= lcm
        d += [{"key": key, "lr": lr}]

        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer, d


def make_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.SCHEDULER == "multi_step":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif cfg.SOLVER.SCHEDULER == "cosine":
        return WarmupCosineAnnealingLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            cfg.SOLVER.ETA_MIN,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
