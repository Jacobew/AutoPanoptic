# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from maskrcnn_benchmark.layers import Conv2d
import torch.nn.functional as F
import numpy as np
import pdb
from PIL import Image
import torchvision
import os
import pycocotools.mask as mask_utils
from itertools import groupby
import numpy as np
import random
import json
from maskrcnn_benchmark.modeling import registry

from torch.nn.parameter import Parameter
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.layers.misc import DFConv2d
from maskrcnn_benchmark.layers import ModulatedDeformConv
from maskrcnn_benchmark.modeling.search_space import head_ss_keys, make_layer
from ..autopanoptic_inter_module import build_inter_module


@registry.SEG_BRANCH.register('AutoPanoptic_Segmentation_Branch_search')
@registry.SEG_BRANCH.register('AutoPanoptic_Segmentation_Branch')
class AutoPanoptic_Segmentation_Branch(torch.nn.Module):
    def __init__(self, cfg, seg_architecture=None, b2f_architecture=None):
        super(AutoPanoptic_Segmentation_Branch, self).__init__()
        self.cfg = cfg.clone()
        assert 'FPN' in cfg.MODEL.BACKBONE.CONV_BODY, 'Segmentation Branch should build on FPN backbone'

        in_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        out_channels = cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL

        self.seg_architecture = None
        self.b2f_architecture = None

        if cfg.NAS.TRAIN_SINGLE_MODEL:
            assert seg_architecture is not None and b2f_architecture is not None
            if cfg.MODEL.SEG_BRANCH.SHARE_SUBNET:
                assert len(seg_architecture) == cfg.MODEL.SEG_BRANCH.SUBNET_DEPTH, 'seg_architecture:{}'.format(len(seg_architecture))
            else:
                assert len(seg_architecture) == cfg.MODEL.SEG_BRANCH.SUBNET_DEPTH * 4, 'seg_architecture:{}'.format(len(seg_architecture))
            assert len(b2f_architecture) == 5
            self.seg_architecture = seg_architecture
            self.b2f_architecture = b2f_architecture

        self.subnets = []
        if cfg.MODEL.SEG_BRANCH.SHARE_SUBNET:
            level = 1
        else:
            level = 4
        for i in range(level):
            subnet_per_layer = []
            for j in range(cfg.MODEL.SEG_BRANCH.SUBNET_DEPTH):
                action_per_depth = nn.ModuleList()
                if self.seg_architecture is None:
                    for k in range(len(head_ss_keys)):
                        if j == 0:
                            action_per_depth.append(make_layer(head_ss_keys[k], cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS, cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL, relu=True, gn=True))
                        else:
                            action_per_depth.append(make_layer(head_ss_keys[k], cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL, cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL, relu=True, gn=True))
                    self.add_module('subnet{}_{}'.format(i, j), action_per_depth)
                    subnet_per_layer.append(action_per_depth)
                else:
                    idx = i * cfg.MODEL.SEG_BRANCH.SUBNET_DEPTH + j
                    if j == 0:
                        action_per_depth = make_layer(head_ss_keys[seg_architecture[idx]], cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS, cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL, relu=True, gn=True)
                    else:
                        action_per_depth = make_layer(head_ss_keys[seg_architecture[idx]], cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL, cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL, relu=True, gn=True)
                    self.add_module('subnet{}_{}'.format(i, j), action_per_depth)
                    subnet_per_layer.append(action_per_depth)
            if self.seg_architecture is None:
                self.subnets.append(subnet_per_layer)
            else:
                self.subnets.append(nn.Sequential(*subnet_per_layer))

        self.score = nn.Conv2d(512, cfg.MODEL.SEG_BRANCH.CLS_NUM, 1)
        self.upsample_rate = 4

        if self.cfg.MODEL.INTER_MODULE.TURN_ON:
            if self.b2f_architecture is None:
                self.b2f = build_inter_module(cfg, cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL * 4, cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS, 5)
            else:
                self.b2f = build_inter_module(cfg, cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL * 4, cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS, 5, architecture=b2f_architecture)
    
    def forward(self, images, features, targets=None, img_ids=None, c2d=None, seg_result=None, ori_sizes=None, rngs=None, forward_0=None, b2f_rngs=None, seg_arguments=None):

        if forward_0 is None or forward_0:
            if self.training:
                assert targets is not None, 'targets should be produced in training segmentation branch'

            else:
                assert img_ids is not None
                assert c2d is not None
                assert seg_result is not None
                assert ori_sizes is not None

            if rngs is not None:
                if self.cfg.MODEL.SEG_BRANCH.SHARE_SUBNET:
                    assert len(rngs) == self.cfg.MODEL.SEG_BRANCH.SUBNET_DEPTH 
                else:
                    assert len(rngs) == self.cfg.MODEL.SEG_BRANCH.SUBNET_DEPTH * 4
                # print('seg:', rngs)

            features = list(features)
            fpn_p2, fpn_p3, fpn_p4, fpn_p5 = features[0], features[1], features[2], features[3]
            if not self.cfg.MODEL.SEG_BRANCH.SHARE_SUBNET:
                if rngs is not None:
                    idx = 0
                    for i in range(4):
                        f = features[i]
                        for d in range(self.cfg.MODEL.SEG_BRANCH.SUBNET_DEPTH):
                            f = self.subnets[i][d][rngs[idx]](f)
                            idx += 1
                        features[i] = f
                    fpn_p2, fpn_p3, fpn_p4, fpn_p5 = features[0], features[1], features[2], features[3]
                else:
                    fpn_p2 = self.subnets[0](fpn_p2)
                    fpn_p3 = self.subnets[1](fpn_p3)
                    fpn_p4 = self.subnets[2](fpn_p4)
                    fpn_p5 = self.subnets[3](fpn_p5)
            else:
                if rngs is not None:
                    idx = 0
                    for i in range(4):
                        f = features[i]
                        for d in range(self.cfg.MODEL.SEG_BRANCH.SUBNET_DEPTH):
                            f = self.subnets[0][d][rngs[idx]](f)
                            idx += 1
                        idx = 0
                        features[i] = f
                    fpn_p2, fpn_p3, fpn_p4, fpn_p5 = features[0], features[1], features[2], features[3]
                else:
                    fpn_p2 = self.subnets[0](fpn_p2)
                    fpn_p3 = self.subnets[0](fpn_p3)
                    fpn_p4 = self.subnets[0](fpn_p4)
                    fpn_p5 = self.subnets[0](fpn_p5)

            fpn_p3 = F.interpolate(fpn_p3, None, 2, mode='bilinear', align_corners=True)
            fpn_p4 = F.interpolate(fpn_p4, None, 4, mode='bilinear', align_corners=True)
            fpn_p5 = F.interpolate(fpn_p5, None, 8, mode='bilinear', align_corners=True)
            
            # feat = torch.cat([fpn_p2, fpn_p3, fpn_p4, fpn_p5], dim=1)
            feat = [fpn_p2, fpn_p3, fpn_p4, fpn_p5]
            # feat = torch.add(fpn_p2, torch.add(fpn_p3, torch.add(fpn_p4, fpn_p5)))

            ##### INTER_MODULE ####
            if self.cfg.MODEL.INTER_MODULE.TURN_ON:
                fpn_attention = self.b2f(feat, rngs=b2f_rngs)
                # print('b2f:', b2f_rngs)

                return {'images': images, 'feat': feat, 'targets': targets, 'ori_sizes': ori_sizes, 'img_ids': img_ids, 'c2d': c2d, 'fpn_attention': fpn_attention}

            #######################



        elif forward_0 is None or forward_0 is False:
            ##### INTER_MODULE ####
            if self.cfg.MODEL.INTER_MODULE.TURN_ON:
                assert seg_arguments is not None
                images = seg_arguments['images']
                feat = seg_arguments['feat']
                targets = seg_arguments['targets']
                ori_sizes = seg_arguments['ori_sizes']
                img_ids = seg_arguments['img_ids']
                c2d = seg_arguments['c2d']
                seg_attention = seg_arguments['seg_attention']
                del seg_arguments
                c = self.cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL

                for i in range(len(seg_attention)):
                    feat[i] = feat[i] + seg_attention[i] * feat[i]

                feat = torch.cat(feat, dim=1)

            #######################

            score = self.score(feat)
            if self.upsample_rate != 1:
                segments = F.interpolate(score, None, self.upsample_rate, mode='bilinear', align_corners=True)

            img_sizes = images.image_sizes
            features_of_origin_size = []
            for i in range(segments.shape[0]):
                features_of_origin_size.append(segments[i, :, :img_sizes[i][0], :img_sizes[i][1]])
                if self.training:
                    assert features_of_origin_size[i].shape[-2:] == targets[i].shape[-2:], \
                                                            'features_of_origin_size.shape:{}, target.shape{}'.format(features_of_origin_size[i].shape, targets[i].shape)

            if not self.training:
                for i in range(len(features_of_origin_size)):
                    features_one_channel = torch.max(features_of_origin_size[i], dim=0)[1]
                    #change to original size
                    features_one_channel = torch.Tensor(np.array(torchvision.transforms.functional.resize(Image.fromarray(np.array(features_one_channel.cpu()).astype('uint8')), ori_sizes[i], Image.NEAREST)))
                    img_id = img_ids[i]
                    if 'coco' in self.cfg.DATASETS.TEST[0]:
                        label_start = 1
                        label_end = 54
                    elif 'ADE' in self.cfg.DATASETS.TEST[0]:
                        label_start = 1
                        label_end = 51
                    for continuous_label in range(label_start, label_end):
                        to_append = dict()
                        mask = (features_one_channel == continuous_label)
                        if not mask.byte().any(): # if there is no mask
                            # print(f'label:{continuous_label} no mask!')
                            continue
                        else:
                            category_id = c2d[continuous_label]
                            to_append['category_id'] = category_id
                            to_append['image_id'] = img_id

                            def binary_mask_to_rle(binary_mask):
                                rle = {'counts': [], 'size': list(binary_mask.shape)}
                                counts = rle.get('counts')
                                for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
                                    if i == 0 and value == 1:
                                        counts.append(0)
                                    counts.append(len(list(elements)))
                                return rle

                            rle = binary_mask_to_rle(np.array(mask.cpu()))
                            rle = mask_utils.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
                            rle['counts'] = rle['counts'].decode('ascii') #byte string is handled differently in Python 2 and Python 3, decode the bytes string to ascii string and dump it in Python 3
                            to_append['segmentation'] = rle
                            seg_result.append(to_append)
                return segments, {}
                
            void_label_num = 0
            for t in targets:
                void_label_num += torch.sum(t.detach().eq(255)).item()

            segment_loss = 0
            for seg_feature_per_img, seg_target_per_img in zip(features_of_origin_size, targets):
                seg_feature_per_img = seg_feature_per_img.unsqueeze(0)
                seg_target_per_img = seg_target_per_img.unsqueeze(0)
                seg_target_per_img = seg_target_per_img.long().cuda()
                Cross_entropy_loss = nn.CrossEntropyLoss(size_average=False,reduction='sum', ignore_index=255) # do not count void label
                # Cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=255) 

                segment_loss = segment_loss + Cross_entropy_loss(seg_feature_per_img, seg_target_per_img)

            pixel_num = sum([f.shape[-2] * f.shape[-1] for f in features_of_origin_size])
            segment_loss = segment_loss / (pixel_num - void_label_num) # normalized by the number of labeled image pixels
            
            return segments, dict(loss_segmentation=segment_loss)

class FCNSubNet(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers, deformable_group=1, dilation=1, with_norm='none'):
        super(FCNSubNet, self).__init__()

        assert with_norm in ['none', 'batch_norm', 'group_norm']
        assert num_layers >= 2
        self.num_layers = num_layers
        if with_norm == 'batch_norm':
            norm = BatchNorm2d
        elif with_norm == 'group_norm':
            def group_norm(in_channel):
                return nn.GroupNorm(32, in_channel)
            norm = group_norm
        else:
            norm = None
        self.conv = nn.ModuleList()
        for i in range(num_layers):
            conv = []
            if i == num_layers - 2:
                conv.append(DFConv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=dilation))
                in_channels = out_channels
            else:
                conv.append(DFConv2d(in_channels, in_channels, kernel_size=3, stride=1, dilation=dilation))
            if with_norm != 'none':
                conv.append(norm(in_channels))
            conv.append(nn.ReLU(inplace=True))
            self.conv.append(nn.Sequential(*conv))

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.fill_(0)
                m.bias.data.fill_(0)
            elif isinstance(m, ModulatedDeformConv):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, x):
        for i in range(self.num_layers):        
            x = self.conv[i](x)
        return x


@registry.SEG_BRANCH.register('UPS_Segmentation_Branch')
class UPS_Segmentation_Branch(torch.nn.Module):
    
    def __init__(self, cfg, adj):
        super(SegmentBranch, self).__init__()
        self.cfg = cfg.clone()
        self.fcn_subnet = FCNSubNet(cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS, 128, 3, with_norm='none')
        self.upsample_rate = 4
        self.score = nn.Conv2d(512, cfg.MODEL.SEG_BRANCH.CLS_NUM, 1)
        # self.score = nn.Conv2d(128, cfg.MODEL.SEG_BRANCH.CLS_NUM, 1)
        
        self.initialize()

    def initialize(self):
        nn.init.normal_(self.score.weight.data, 0, 0.01)
        self.score.bias.data.zero_()


    def forward(self, images, features, targets=None, img_ids=None, c2d=None, seg_result=None, ori_sizes=None):
        if self.training:
            assert targets is not None, 'targets should be produced in training segmentation branch'

        else:
            assert img_ids is not None
            assert c2d is not None
            assert seg_result is not None
            assert ori_sizes is not None

        fpn_p2, fpn_p3, fpn_p4, fpn_p5 = features[0], features[1], features[2], features[3]
        fpn_p2 = self.fcn_subnet(fpn_p2)
        fpn_p3 = self.fcn_subnet(fpn_p3)
        fpn_p4 = self.fcn_subnet(fpn_p4)
        fpn_p5 = self.fcn_subnet(fpn_p5)

        fpn_p3 = F.interpolate(fpn_p3, None, 2, mode='bilinear', align_corners=True)
        fpn_p4 = F.interpolate(fpn_p4, None, 4, mode='bilinear', align_corners=True)
        fpn_p5 = F.interpolate(fpn_p5, None, 8, mode='bilinear', align_corners=True)
        
        feat = torch.cat([fpn_p2, fpn_p3, fpn_p4, fpn_p5], dim=1)
        # feat = torch.add(fpn_p2, torch.add(fpn_p3, torch.add(fpn_p4, fpn_p5)))
        score = self.score(feat)
        if self.upsample_rate != 1:
            segments = F.interpolate(score, None, self.upsample_rate, mode='bilinear', align_corners=True)

        img_sizes = images.image_sizes
        features_of_origin_size = []
        for i in range(segments.shape[0]):
            features_of_origin_size.append(segments[i, :, :img_sizes[i][0], :img_sizes[i][1]])
            if self.training:
                assert features_of_origin_size[i].shape[-2:] == targets[i].shape[-2:], \
                                                        'features_of_origin_size.shape:{}, target.shape{}'.format(features_of_origin_size[i].shape, targets[i].shape)

        if not self.training:
            for i in range(len(features_of_origin_size)):
                features_one_channel = torch.max(features_of_origin_size[i], dim=0)[1]
                #change to original size
                features_one_channel = torch.Tensor(np.array(torchvision.transforms.functional.resize(Image.fromarray(np.array(features_one_channel.cpu()).astype('uint8')), ori_sizes[i], Image.NEAREST)))
                img_id = img_ids[i]
                if 'coco' in self.cfg.DATASETS.TEST[0]:
                    label_start = 1
                    label_end = 54
                elif 'ADE' in self.cfg.DATASETS.TEST[0]:
                        label_start = 1
                        label_end = 51
                for continuous_label in range(label_start, label_end):
                    to_append = dict()
                    mask = (features_one_channel == continuous_label)
                    if not mask.byte().any(): # if there is no mask
                        continue
                    else:
                        category_id = c2d[continuous_label]
                        to_append['category_id'] = category_id
                        to_append['image_id'] = img_id

                        def binary_mask_to_rle(binary_mask):
                            rle = {'counts': [], 'size': list(binary_mask.shape)}
                            counts = rle.get('counts')
                            for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
                                if i == 0 and value == 1:
                                    counts.append(0)
                                counts.append(len(list(elements)))
                            return rle

                        rle = binary_mask_to_rle(np.array(mask.cpu()))
                        rle = mask_utils.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
                        rle['counts'] = rle['counts'].decode('ascii') #byte string is handled differently in Python 2 and Python 3, decode the bytes string to ascii string and dump it in Python 3
                        to_append['segmentation'] = rle
                        seg_result.append(to_append)
            return segments, {}
            
        void_label_num = 0
        for t in targets:
            void_label_num += torch.sum(t.detach().eq(255)).item()

        segment_loss = 0
        for seg_feature_per_img, seg_target_per_img in zip(features_of_origin_size, targets):
            seg_feature_per_img = seg_feature_per_img.unsqueeze(0)
            seg_target_per_img = seg_target_per_img.unsqueeze(0)
            seg_target_per_img = seg_target_per_img.long().cuda()
            Cross_entropy_loss = nn.CrossEntropyLoss(size_average=False,reduction='sum', ignore_index=255) # do not count void label
            # Cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=255) 

            segment_loss = segment_loss + Cross_entropy_loss(seg_feature_per_img, seg_target_per_img)

        pixel_num = sum([f.shape[-2] * f.shape[-1] for f in features_of_origin_size])
        segment_loss = segment_loss / (pixel_num - void_label_num) # normalized by the number of labeled image pixels
        
        return segments, dict(loss_segmentation=segment_loss)


@registry.SEG_BRANCH.register('Panoptic_FPN_Segmentation_Branch')
class Panoptic_FPN_Segmentation_Branch(torch.nn.Module):
    
    def __init__(self, cfg):
        super(Panoptic_FPN_Segmentation_Branch, self).__init__()
        self.cfg = cfg.clone()
        assert 'FPN' in cfg.MODEL.BACKBONE.CONV_BODY, 'Segmentation Branch should build on FPN backbone'
        # Resnet backbone has 4 stages
        self.upsample_level1 = nn.Sequential(
                                Conv2d(cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS, cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL, 3, 1, 1),
                                nn.GroupNorm(num_groups=32, num_channels=cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL),
                                nn.ReLU(True)
                             )

        self.upsample_level2 = nn.Sequential(
                                Conv2d(cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS, cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL, 3, 1, 1),
                                nn.GroupNorm(num_groups=32, num_channels=cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL),
                                nn.ReLU(True),
                                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
                             )
        
        self.upsample_level3 = nn.Sequential(
                                Conv2d(cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS, cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL, 3, 1, 1),
                                nn.GroupNorm(num_groups=32, num_channels=cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL),
                                nn.ReLU(True),
                                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
                                Conv2d(cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL, cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL, 3, 1, 1),
                                nn.GroupNorm(num_groups=32, num_channels=cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL),
                                nn.ReLU(True),
                                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
                             )
        
        self.upsample_level4 = nn.Sequential(
                                Conv2d(cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS, cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL, 3, 1, 1),
                                nn.GroupNorm(num_groups=32, num_channels=cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL),
                                nn.ReLU(True),
                                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
                                Conv2d(cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL, cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL, 3, 1, 1),
                                nn.GroupNorm(num_groups=32, num_channels=cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL),
                                nn.ReLU(True),
                                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
                                Conv2d(cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL, cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL, 3, 1, 1),
                                nn.GroupNorm(num_groups=32, num_channels=cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL),
                                nn.ReLU(True),
                                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
                             )

        if cfg.MODEL.SEG_BRANCH.MERGE_OP == "add": 
            self.to_segment_conv = Conv2d(cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL, cfg.MODEL.SEG_BRANCH.CLS_NUM, 1, 1)
        else:
            self.to_segment_conv = Conv2d(cfg.MODEL.SEG_BRANCH.DECODER_CHANNEL * 4, cfg.MODEL.SEG_BRANCH.CLS_NUM, 1, 1)

        self.to_segment_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        

    def reset_param(self):
        for para in self.parameters():
            torch.nn.init.xavier_uniform_(para)


    def forward(self, images, features, targets=None, img_ids=None, c2d=None, seg_result=None, ori_sizes=None):

        if self.training:
            assert targets is not None, 'targets should be produced in training segmentation branch'

        else:
            assert img_ids is not None
            assert c2d is not None
            assert seg_result is not None
            assert ori_sizes is not None
               
        features_level1_upsampled = self.upsample_level1(features[0])
        features_level2_upsampled = self.upsample_level2(features[1])
        features_level3_upsampled = self.upsample_level3(features[2])
        features_level4_upsampled = self.upsample_level4(features[3])

        if self.cfg.MODEL.SEG_BRANCH.MERGE_OP == "add":
            merge_features = torch.add(features_level1_upsampled, torch.add(features_level2_upsampled, torch.add(features_level3_upsampled, features_level4_upsampled)))
        elif self.cfg.MODEL.SEG_BRANCH.MERGE_OP == "merge":
            merge_features = torch.cat([features_level1_upsampled, features_level2_upsampled, features_level3_upsampled, features_level4_upsampled], dim=1)

        img_sizes = images.image_sizes
        segments = self.to_segment_conv(merge_features) # segments: BS x C x H x W
        segments = self.to_segment_upsample(segments)

        features_of_origin_size = []
        for i in range(segments.shape[0]):
            features_of_origin_size.append(segments[i, :, :img_sizes[i][0], :img_sizes[i][1]])
            if self.training:
                assert features_of_origin_size[i].shape[-2:] == targets[i].shape[-2:], \
                                                        'features_of_origin_size.shape:{}, target.shape{}'.format(features_of_origin_size[i].shape, targets[i].shape)

        if not self.training:
            for i in range(len(features_of_origin_size)):
                features_one_channel = torch.max(features_of_origin_size[i], dim=0)[1]
                #change to original size
                features_one_channel = torch.Tensor(np.array(torchvision.transforms.functional.resize(Image.fromarray(np.array(features_one_channel.cpu()).astype('uint8')), ori_sizes[i], Image.NEAREST)))
                img_id = img_ids[i]
                if 'coco' in self.cfg.DATASETS.TEST[0]:
                    label_start = 1
                    label_end = 54
                elif 'ADE' in self.cfg.DATASETS.TEST[0]:
                    label_start = 1
                    label_end = 51
                for continuous_label in range(label_start, label_end):
                    to_append = dict()
                    mask = (features_one_channel == continuous_label)
                    if not mask.byte().any(): # if there is no mask
                        continue
                    else:
                        category_id = c2d[continuous_label]
                        to_append['category_id'] = category_id
                        to_append['image_id'] = img_id

                        def binary_mask_to_rle(binary_mask):
                            rle = {'counts': [], 'size': list(binary_mask.shape)}
                            counts = rle.get('counts')
                            for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
                                if i == 0 and value == 1:
                                    counts.append(0)
                                counts.append(len(list(elements)))
                            return rle

                        rle = binary_mask_to_rle(np.array(mask.cpu()))
                        rle = mask_utils.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
                        rle['counts'] = rle['counts'].decode('ascii') #byte string is handled differently in Python 2 and Python 3, decode the bytes string to ascii string and dump it in Python 3
                        to_append['segmentation'] = rle
                        seg_result.append(to_append)
            return segments, {}

        void_label_num = 0
        for t in targets:
            void_label_num += torch.sum(t.detach().eq(255)).item()

        segment_loss = 0
        for seg_feature_per_img, seg_target_per_img in zip(features_of_origin_size, targets):
            seg_feature_per_img = seg_feature_per_img.unsqueeze(0)
            seg_target_per_img = seg_target_per_img.unsqueeze(0)
            seg_target_per_img = seg_target_per_img.long().cuda()
            Cross_entropy_loss = nn.CrossEntropyLoss(size_average=False,reduction='sum', ignore_index=255) # do not count void label
            # Cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=255) 

            segment_loss = segment_loss + Cross_entropy_loss(seg_feature_per_img, seg_target_per_img)

        pixel_num = sum([f.shape[-2] * f.shape[-1] for f in features_of_origin_size])
        segment_loss = segment_loss / (pixel_num - void_label_num) # normalized by the number of labeled image pixels

        return segments, dict(loss_segmentation=segment_loss)


def build_seg_branch(cfg, seg_architecture=None, b2f_architecture=None):
    func = registry.SEG_BRANCH[
        cfg.MODEL.SEG_BRANCH.SEGMENT_BRANCH
    ]
    if 'AutoPanoptic' in cfg.MODEL.SEG_BRANCH.SEGMENT_BRANCH:
        return func(cfg, seg_architecture=seg_architecture, b2f_architecture=b2f_architecture)
    return func(cfg)

if __name__ == '__main__':
    from maskrcnn_benchmark.config import cfg
    seg_branch = build_seg_branch(cfg)
    print(seg_branch)
