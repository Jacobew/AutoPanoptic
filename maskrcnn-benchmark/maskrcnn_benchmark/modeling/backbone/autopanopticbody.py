import torch.nn as nn
from maskrcnn_benchmark.modeling.backbone.search_space import ConvBNReLU, ShuffleNetV2BlockSearched, blocks_key

class AutoPanopticBody(nn.Module):
	def __init__(self, cfg, architecture=None):
		super(AutoPanopticBody, self).__init__()

		# [-1, 48, 96, 240, 480, 960]  
		stage_out_channels = cfg.MODEL.BACKBONE.STAGE_OUT_CHANNELS
		# [8, 8, 16, 8]  
		stage_repeats = cfg.MODEL.BACKBONE.STAGE_REPEATS

		self.architecture = architecture

		if 'search' in cfg.MODEL.BACKBONE.CONV_BODY:
			assert architecture is None
			self.blocks_key = blocks_key
			self.num_states = sum(stage_repeats)
		else:
			assert architecture is not None
			self.architecture = architecture

		self.first_conv = ConvBNReLU(in_channel=3, out_channel=stage_out_channels[1], k_size=3, stride=2, padding=1, gaussian_init=True)

		self.features = list()
		self.stage_ends_idx = list()

		in_channels = stage_out_channels[1]
		i_th = 0
		# build backbone
		for id_stage in range(1, len(stage_repeats) + 1):
			out_channels = stage_out_channels[id_stage + 1]
			repeats = stage_repeats[id_stage - 1]
			# build each stage
			for id_repeat in range(repeats):
				prefix = str(id_stage) + chr(ord('a') + id_repeat)
				stride = 1 if id_repeat > 0 else 2 # downsample at the first layer in each stage
				if architecture is None:
					_ops = nn.ModuleList()
					# build all path in search space
					for i in range(len(blocks_key)): 
						_ops.append(ShuffleNetV2BlockSearched(prefix, in_channels=in_channels, out_channels=out_channels,
                                                               stride=stride, base_mid_channels=out_channels // 2, id=i))
					self.features.append(_ops)
				else:
					self.features.append(ShuffleNetV2BlockSearched(prefix, in_channels=in_channels, out_channels=out_channels,
                                                               stride=stride, base_mid_channels=out_channels // 2, id=architecture[i_th]))

				in_channels = out_channels
				i_th += 1
			self.stage_ends_idx.append(i_th-1)

		self.features = nn.Sequential(*self.features)

	def forward(self, x, rngs=None):
		outputs = []
		x = self.first_conv(x)
		for i, select_op in enumerate(self.features):
			x = select_op(x) if rngs is None else select_op[rngs[i]](x)
			if i in self.stage_ends_idx:
				outputs.append(x)
		return outputs

if __name__ == "__main__":
	from maskrcnn_benchmark.config import cfg
	model = AutoPanopticBody(cfg)
	print(model)







































