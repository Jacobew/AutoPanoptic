# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3

from maskrcnn_benchmark.modeling.search_space import head_ss_keys, make_layer


registry.ROI_MASK_FEATURE_EXTRACTORS.register(
    "ResNet50Conv5ROIFeatureExtractor", ResNet50Conv5ROIFeatureExtractor
)

@registry.ROI_MASK_FEATURE_EXTRACTORS.register("AutoPanoptic_MaskRCNNFPNFeatureExtractor_search")
@registry.ROI_MASK_FEATURE_EXTRACTORS.register("AutoPanoptic_MaskRCNNFPNFeatureExtractor")
class AutoPanoptic_MaskRCNNFPNFeatureExtractor(nn.Module):
    def __init__(self, cfg, in_channels, architecture=None):
        super(AutoPanoptic_MaskRCNNFPNFeatureExtractor, self).__init__()
        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION

        next_feature = input_size
        self.blocks = []

        self.architecture = None

        if 'AutoPanoptic' in cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR and \
            'search' not in cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR:
            assert architecture is not None, 'architecture not specified in AutoPanoptic mask head'
            assert len(architecture) == len(cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS)
            self.architecture = architecture

        for layer_idx, layer_features in enumerate(layers, 1):
            if architecture is None:
                _ops = nn.ModuleList()
                for i in range(len(head_ss_keys)):
                    _ops.append(make_layer(head_ss_keys[i], next_feature, layer_features, relu=False, gn=True))
                next_feature = layer_features
                self.blocks.append(_ops)
            else:
                _ops = make_layer(head_ss_keys[architecture[layer_idx-1]], next_feature, layer_features, relu=False, gn=True)
                next_feature = layer_features
                self.blocks.append(_ops)
            self.add_module('AutoPanoptic_mask_fcn_{}'.format(layer_idx), _ops) # inconsistent module name between search and single model can incur problem in model reloading

        self.out_channels = layer_features

    def forward(self, x, proposals, rngs=None):
        if rngs is not None:
            assert len(rngs) == len(self.blocks)
            assert self.architecture is None
        # print('mask:', rngs)
        x = self.pooler(x, proposals)

        for i, block in enumerate(self.blocks):
            x = F.relu(block(x) if rngs is None else block[rngs[i]](x))

        return x


@registry.ROI_MASK_FEATURE_EXTRACTORS.register("MaskRCNNFPNFeatureExtractor")
class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1, use_gn=use_gn
            )
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x


def make_roi_mask_feature_extractor(cfg, in_channels, architecture=None):
    func = registry.ROI_MASK_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR
    ]
    if 'AutoPanoptic' in cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR:
        return func(cfg, in_channels, architecture=architecture)
    return func(cfg, in_channels)

if __name__ == "__main__":
    from maskrcnn_benchmark.config import cfg
    mask_head = AutoPanoptic_MaskRCNNFPNFeatureExtractor(cfg, 256)
    print(mask_head)
