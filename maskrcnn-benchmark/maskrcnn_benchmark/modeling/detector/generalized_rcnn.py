# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..seg_branch.seg_branch import build_seg_branch
from ..search_space import head_ss_keys
from ..autopanoptic_inter_module import build_inter_module
from maskrcnn_benchmark.modeling.search_space import make_inter_layer, inter_ss_keys
import torch.nn.functional as F
from maskrcnn_benchmark.utils.comm import get_world_size, synchronize, get_rank


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg, model_config=None):
        super(GeneralizedRCNN, self).__init__()

        self.cfg = cfg.clone()
        self.backbone_cfg = model_config.get('backbone') if model_config is not None else None
        self.head_cfg = model_config.get('head') if model_config is not None else None
        self.inter_cfg = model_config.get('inter') if model_config is not None else None

        if 'search' in cfg.MODEL.BACKBONE.CONV_BODY:
            assert self.backbone_cfg is None
            self.backbone, self.fpn = build_backbone(cfg)
            self.blocks_key = self.backbone.blocks_key
            self.num_states = self.backbone.num_states
        else:
            # assert self.backbone_cfg is not None
            self.backbone = build_backbone(cfg, architecture=self.backbone_cfg)

        self.head_ss_keys = head_ss_keys
        out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS

        self.rpn = build_rpn(cfg, out_channels)

        mask_cfg = self.head_cfg[ :len(cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS)] if self.head_cfg is not None else None
        seg_cfg = self.head_cfg[len(cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS): ] if self.head_cfg is not None else None
        f2b_cfg = self.inter_cfg[: -5] if self.inter_cfg is not None else None
        b2f_cfg = self.inter_cfg[-5: ] if self.inter_cfg is not None else None
        self.roi_heads = build_roi_heads(cfg, out_channels, architecture=mask_cfg)
        self.seg_branch = build_seg_branch(cfg, seg_architecture=seg_cfg, b2f_architecture=b2f_cfg)

        if cfg.MODEL.INTER_MODULE.TURN_ON:
            self.f2b = build_inter_module(cfg, cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS * 4, cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL, 4, architecture=f2b_cfg)

    def forward(self, images, targets=None, segment_target=None, img_ids=None, c2d=None, seg_result=None, ori_sizes=None, rngs=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        backbone_rngs, mask_rngs, seg_rngs, f2b_rngs, b2f_rngs = None, None, None, None, None
        if self.cfg.NAS.TRAIN_SUPERNET:
            _cur = 0
            if 'search' in self.cfg.MODEL.BACKBONE.CONV_BODY:
                backbone_rngs = rngs[_cur : _cur+self.num_states]
                _cur += self.num_states
            mask_rngs = rngs[_cur : _cur+len(self.cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS)]
            _cur += len(self.cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS)
            seg_rngs = rngs[_cur: -9]
            f2b_rngs = rngs[-9: -5]
            b2f_rngs = rngs[-5: ]
            # print('rngs:', backbone_rngs, mask_rngs, seg_rngs, f2b_rngs, b2f_rngs)

        if rngs is None: # training single model
            assert not self.cfg.NAS.TRAIN_SUPERNET
            features = self.backbone(images.tensors)
        else: # TRAIN_SUPERNET is True
            if backbone_rngs is not None: # training supernet
                features = self.backbone(images.tensors, backbone_rngs)
                features = self.fpn(features)
            else: # no backbone search
                features = self.backbone(images.tensors)
                

        ##### INTER MODULE #####
        if self.cfg.MODEL.INTER_MODULE.TURN_ON:
            rpn_upsample_f = []
            for i in range(-1, -len(features), -1):
                rpn_upsample_f.insert(0, F.interpolate(features[i], size=features[0].size()[-2:], mode='bilinear', align_corners=True))
            rpn_upsample_f = torch.cat(rpn_upsample_f, dim=1)

            seg_attention = self.f2b(rpn_upsample_f, f2b_rngs)
            # print('f2b:', f2b_rngs)

            if self.cfg.MODEL.SEG_BRANCH.ADD_SEG_BRANCH: 
                seg_arguments = self.seg_branch(images, features, segment_target, img_ids=img_ids, c2d=c2d, seg_result=seg_result, ori_sizes=ori_sizes, rngs=seg_rngs, forward_0=True, b2f_rngs=b2f_rngs)
                seg_arguments['seg_attention'] = seg_attention

            features = list(features)
            fpn_attention = seg_arguments['fpn_attention'] # bs x 256 x 1 x 1
            assert len(fpn_attention) == len(features)
            for i in range(len(features)):
                features[i] = features[i] + fpn_attention[i] * features[i]

        ########################
        
        proposals, proposal_losses = self.rpn(images, features, targets)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets, mask_rngs)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.cfg.MODEL.SEG_BRANCH.ADD_SEG_BRANCH:
            if self.cfg.MODEL.INTER_MODULE.TURN_ON: 
                segments, segmentation_loss = self.seg_branch(images, features, segment_target, img_ids=img_ids, c2d=c2d, seg_result=seg_result, ori_sizes=ori_sizes, rngs=seg_rngs, forward_0=False, b2f_rngs=b2f_rngs, seg_arguments=seg_arguments)
            else:
                segments, segmentation_loss = self.seg_branch(images, features, segment_target, img_ids=img_ids, c2d=c2d, seg_result=seg_result, ori_sizes=ori_sizes, rngs=seg_rngs, b2f_rngs=b2f_rngs)

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            if self.cfg.MODEL.SEG_BRANCH.ADD_SEG_BRANCH:
                losses.update(segmentation_loss) #
            return losses

        return result
