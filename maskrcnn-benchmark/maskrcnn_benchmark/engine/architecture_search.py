import torch
import random
import math
from collections import namedtuple
import numpy as np
import json
import os
from datetime import datetime
import time
from maskrcnn_benchmark.modeling.search_space import head_ss_keys, inter_ss_keys
from maskrcnn_benchmark.modeling.backbone.search_space import blocks_key
from maskrcnn_benchmark.engine.trainer import generate_rng
from maskrcnn_benchmark.utils.comm import get_world_size, synchronize, get_rank
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.data import make_data_loader
import torch.distributed as dist


class TEST_MODEL(object):
	def __init__(self, backbone_rngs, head_rngs, inter_rngs, cycle, \
				pq=None, pq_thing=None, pq_stuff=None, sq=None, rq=None, \
				ap_box=None, ap_mask=None, fitness=None, fitness_key="pq"):
		self.backbone_rngs = backbone_rngs
		self.head_rngs = head_rngs
		self.inter_rngs = inter_rngs

		self.backbone_layers = len(backbone_rngs) if backbone_rngs is not None else 0
		self.head_layers = len(head_rngs)
		self.inter_layers = len(inter_rngs)

		self.fitness = fitness
		self.pq = pq
		self.pq_thing = pq_thing
		self.pq_stuff = pq_stuff
		self.sq = sq
		self.rq = rq
		self.ap_box = ap_box
		self.ap_mask = ap_mask
		self.cycle = cycle

		self.fitness_key = fitness_key

	def set_fitness(self, fitness):
		self.fitness = float(fitness)

	def set(self, pq, pq_thing, pq_stuff, sq, rq, ap_box, ap_mask):
		self.pq = pq
		self.pq_thing = pq_thing
		self.pq_stuff = pq_stuff
		self.sq = sq
		self.rq = rq
		self.ap_box = ap_box
		self.ap_mask = ap_mask

	@staticmethod
	def reload(l, fitness_key="pq"):
		assert isinstance(l, dict)
		loaded_model = TEST_MODEL(l["backbone_rngs"], l['head_rngs'], l['inter_rngs'], l['cycle'])
		loaded_model.set(l['pq'], l['pq_thing'], l['pq_stuff'], l['sq'], l['rq'], l['ap_box'], l['ap_mask'])
		loaded_model.set_fitness(l[fitness_key])
		return loaded_model

	def __repr__(self):
		info = "TEST_MODEL: [backbone_rngs:{}, head_rngs:{}, inter_rngs:{}, cycle:{}".format(self.backbone_rngs, self.head_rngs, self.inter_rngs, self.cycle)
		if self.pq is not None:
			info += ", pq:{}, pq_thing:{}, pq_stuff:{}, sq:{}, rq:{}, ap_box:{}, ap_mask:{}]".format(
				self.pq, self.pq_thing, self.pq_stuff, self.sq, self.rq, self.ap_box, self.ap_mask)
		else:
			info += "]"
		return info

	def encode(self): # for broadcast
		enc = []
		if self.backbone_rngs is not None:
			enc.extend(self.backbone_rngs)
		enc.extend(self.head_rngs)
		enc.extend(self.inter_rngs)
		enc.append(self.cycle)
		return torch.Tensor(enc)

	def tolist(self):
		l = []
		if self.backbone_rngs is not None:
			l.extend(self.backbone_rngs)
		l.extend(self.head_rngs)
		l.extend(self.inter_rngs)
		return l, self.cycle

	@staticmethod
	def fromlist(l, split):
		assert isinstance(split, list)
		for i in range(1, len(split)):
			split[i] += split[i-1]
		temp = np.split(l, split)
		map(lambda x: x.tolist(), temp)
		assert len(temp[-1]) == 1
		temp[-1] = temp[-1][0]
		if len(split) < 3: # no backbone search
			temp.insert(0, [])
		return temp

	def info(self):
		def serialize(_input):
			if not isinstance(_input, list):
				_input = _input.tolist()
			for i in range(len(_input)):
				_input[i] = int(_input[i])
			return _input
		backbone_rngs = serialize(self.backbone_rngs)
		head_rngs = serialize(self.head_rngs)
		inter_rngs = serialize(self.inter_rngs)
		return dict(
				{
					'cycle': int(self.cycle),
					'backbone_rngs': backbone_rngs,
					'head_rngs': head_rngs,
					'inter_rngs': inter_rngs,
					'pq': float(self.pq),
					'pq_thing': float(self.pq_thing),
					'pq_stuff': float(self.pq_stuff),
					'sq': float(self.sq),
					'rq': float(self.rq),
					'ap_box': float(self.ap_box),
					'ap_mask': float(self.ap_mask),
					'fitness': float(self.fitness)
				}
			   )

class PathPrioritySearch(object):
	'''
		Distributed Path Priority Search 
	'''
	def __init__(self, cfg, base_dir, topk=5, ckp="cache_test_log.json"):
		self.cfg = cfg.clone()
		self.ckp = ckp
		self.topk = topk
		self.base_dir = base_dir
		if not os.path.exists(base_dir):
			os.mkdir(base_dir)

		self.num_cycle = self.cfg.NAS.TEST_CYCLE

		self.backbone_layers = sum(self.cfg.MODEL.BACKBONE.STAGE_REPEATS)
		if self.cfg.MODEL.SEG_BRANCH.SHARE_SUBNET:
			self.head_layers = len(cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS) + cfg.MODEL.SEG_BRANCH.SUBNET_DEPTH
		else:
			self.head_layers = len(cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS) + 4 * cfg.MODEL.SEG_BRANCH.SUBNET_DEPTH
		self.inter_layers = 9

		self.backbone_ss_size = len(blocks_key)
		self.head_ss_size = len(head_ss_keys)
		self.inter_ss_size = len(inter_ss_keys)

		if 'search' in cfg.MODEL.BACKBONE.CONV_BODY:
			_lcm = self.backbone_ss_size*self.head_ss_size//math.gcd(self.backbone_ss_size,self.head_ss_size)
			self.lcm = self.inter_ss_size*_lcm//math.gcd(self.inter_ss_size, _lcm)
			self.search_backbone = True
		else:
			self.lcm = self.inter_ss_size*self.head_ss_size//math.gcd(self.inter_ss_size, self.head_ss_size)
			self.search_backbone = False

		self.cache_model_list = None # type: list[TEST_MODEL]
		self.exist_cycle = -1
		self.new_model_list = None # type: list[TEST_MODEL]

		self.backbone_sb = np.zeros((self.backbone_ss_size, self.backbone_layers), dtype=np.int)
		self.head_sb = np.zeros((self.head_ss_size, self.head_layers), dtype=np.int)
		self.inter_sb = np.zeros((self.inter_ss_size, self.inter_layers), dtype=np.int)

		self.rank = get_rank()
		self.world_size = get_world_size()


	def generate_fair_test(self): # load cache model and generate new models to test
		cache_f = self.load_checkpoint()
		cache_model_list = []
		new_model_list = []
		
		exist_cycle = -1
		# load evluated results
		if cache_f is not None:
			print('length of cache:', len(cache_f))
			exist_cycle = len(cache_f) // self.lcm
			print('existing cycles:', exist_cycle)
			cache_f = cache_f[0: self.lcm * exist_cycle]
			for m in cache_f:
				# t = TEST_MODEL(m['backbone_rngs'], m['head_rngs'], m['inter_rngs'], m['cycle'])
				# t.set(m['pq'], m['pq_thing'], m['pq_stuff'], m['sq'], m['rq'], m['ap_box'], m['ap_mask'])
				# t.set_fitness(m['pq'])
				t = TEST_MODEL(m.get('backbone_rngs'), m.get('head_rngs'), m.get('inter_rngs'), m.get('cycle'))
				t.set(m.get('pq'), m.get('pq_thing'), m.get('pq_stuff'), m.get('sq'), m.get('rq'), m.get('ap_box'), m.get('ap_mask'))
				t.set_fitness(m.get('pq'))
				cache_model_list.append(t)
			print('Loaded cache models:', cache_model_list)
		else:
			exist_cycle = 0

		# generate new models to evaluate
		new_model_enc = None
		
		for c in range(exist_cycle, self.num_cycle):
			if self.search_backbone:
				backbone_rngs = generate_rng(self.backbone_layers, self.backbone_ss_size, self.lcm).transpose(1, 0)
			head_rngs = generate_rng(self.head_layers, self.head_ss_size, self.lcm).transpose(1, 0)
			inter_rngs = generate_rng(self.inter_layers, self.inter_ss_size, self.lcm).transpose(1, 0)
			# rngs = np.concatenate([backbone_rngs, head_rngs, inter_rngs], axis=0).transpose(1, 0)
			
			for i in range(len(head_rngs)):
				if self.search_backbone:
					new_model_list.append(TEST_MODEL(backbone_rngs[i], head_rngs[i], inter_rngs[i], c))
				else:
					new_model_list.append(TEST_MODEL(None, head_rngs[i], inter_rngs[i], c))
		# if self.rank == 0:
		# 	print('rank 0:', new_model_list)

		if exist_cycle < self.num_cycle: 
			new_model_enc = torch.stack([m.encode() for m in new_model_list]).cuda().detach()
				
			if self.world_size > 1:
				dist.broadcast(new_model_enc, 0)
			new_model_list = new_model_enc.cpu().numpy().astype(np.int).tolist()
			# convert list to TEST_MODEL
			for i in range(len(new_model_list)):
				if self.search_backbone:
					temp = TEST_MODEL.fromlist(new_model_list[i], [self.backbone_layers, self.head_layers, self.inter_layers])
				else:
					temp = TEST_MODEL.fromlist(new_model_list[i], [self.head_layers, self.inter_layers])
				new_model_list[i] = TEST_MODEL(*temp)
		# if self.rank == 1:
		# 	print('Receive:', new_model_list)

		assert(len(cache_model_list) == self.lcm * exist_cycle), "len(cache_model_list)={}, lcm * exist_cycle={}, cache_model_list:{}".format(len(cache_model_list), self.lcm * exist_cycle, cache_model_list)
		self.cache_model_list = cache_model_list
		self.new_model_list = new_model_list
		self.exist_cycle = exist_cycle


	def score_cycle_model(self, cycle_models, verbose=True):
		if self.rank == 0: # only rank 0 has results to score
			assert len(cycle_models) == self.lcm, 'models missing in some cycle'
			assert len(set([m.cycle for m in cycle_models])) == 1, 'models are not in a same cycle:\n{}'.format(cycle_models)
			cur_score = self.lcm - 1
			model_list = sorted(cycle_models, key=lambda item: item.fitness, reverse=True)
			for model in model_list:
				if self.search_backbone:
					self.backbone_sb[model.backbone_rngs, np.arange(self.backbone_layers)] += cur_score
				self.head_sb[model.head_rngs, np.arange(self.head_layers)] += cur_score
				self.inter_sb[model.inter_rngs, np.arange(self.inter_layers)] += cur_score
				cur_score -= 1
			if verbose:
				self.myprint('='*15, ' Scoreboard ', '='*15)
				self.myprint('backbone:', self.backbone_sb)
				self.myprint('head:', self.head_sb)
				self.myprint('inter:', self.inter_sb)


	def save_checkpoint(self, file_name):
		if self.rank == 0:
			output = []
			fn = os.path.join(self.base_dir, file_name)
			for m in self.cache_model_list:
				output.append(m.info())
			print("Saving checkpoint to {}".format(fn))
			with open(fn, 'w') as f:
				json.dump(output, f)
			sb = {}
			if self.search_backbone:
				sb['backbone_sb'] = self.backbone_sb.tolist()
			sb['head_sb'] = self.head_sb.tolist()
			sb['inter_sb'] = self.inter_sb.tolist()
			sb_fn = fn.rstrip('.json') + '_scoreboard.json'
			print("Saving scoreboard to {}".format(sb_fn))
			with open(sb_fn, 'w') as f:
				json.dump(sb, f)


	def load_checkpoint(self):
		cache_f = None
		ckp_fn = os.path.join(self.base_dir, self.ckp)
		if os.path.exists(ckp_fn):
			print("Using (part of) cached test results from {}".format(ckp_fn))
			cache_f = json.load(open(ckp_fn, 'r'))
		return cache_f


	def save_topk(self):
		if self.rank == 0:
			d = 'top{}_cycle{}_{}'.format(self.topk, self.num_cycle, datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M'))
			save_dir = os.path.join(self.base_dir, d)
			if not os.path.exists(save_dir):
				os.mkdir(save_dir)
			# reverse argsort
			if self.search_backbone:
				backbone_topk = np.argsort(self.backbone_sb, axis=0)[-self.topk:][::-1].tolist()
			head_topk = np.argsort(self.head_sb, axis=0)[-self.topk:][::-1].tolist()
			inter_topk = np.argsort(self.inter_sb, axis=0)[-self.topk:][::-1].tolist()
			def get_min(*a):
			    ans = a[0]
			    for item in a:
			        ans = min(ans, item)
			    return ans
			if self.search_backbone:
				_min = get_min(self.topk, self.backbone_ss_size, self.head_ss_size, self.inter_ss_size)
			else:
				_min = get_min(self.topk, self.head_ss_size, self.inter_ss_size)
			for i in range(_min):
				with open(os.path.join(save_dir, 'model_{}'.format(i)), 'w') as f:
					print('***'*10)
					print('Saving model_{}'.format(i))
					if self.search_backbone:
						print('backbone:', backbone_topk[i])
					print('head:', head_topk[i])
					print('inter:', inter_topk[i])
					if self.search_backbone:
						json.dump(
							[{
								'backbone': backbone_topk[i],
								'head': head_topk[i],
								'inter': inter_topk[i],
							}], f
						)
					else:
						json.dump(
							[{
								'head': head_topk[i],
								'inter': inter_topk[i],
							}], f
						)


	def _evaluate(self, model, model_cfg, data_loaders, output_folders, dataset_names):		
		iou_types = ("bbox",)
		if self.cfg.MODEL.MASK_ON:
			iou_types = iou_types + ("segm",)
		rngs, _ = model_cfg.tolist()
		for output_folder, dataset_name, data_loader in zip(output_folders, dataset_names, data_loaders):
			temp = inference(
							model,
							data_loader,
							dataset_name=dataset_name,
							iou_types=iou_types,
							box_only=self.cfg.MODEL.RPN_ONLY,
							device=self.cfg.MODEL.DEVICE,
							expected_results=self.cfg.TEST.EXPECTED_RESULTS,
							expected_results_sigma_tol=self.cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
							output_folder=output_folder,
							c2d_json_path=self.cfg.MODEL.SEG_BRANCH.JSON_PATH,
							rngs=rngs,
							cfg=self.cfg,
						)
			synchronize()

			def _list2str(lst):
				return "".join(map(lambda x: str(x), lst))

			def _parse(f):
				# return int(np.random.randint(0, 100)), 0, 0, 0, 0
				if isinstance(f, str):
					f = open(f, 'r')
				for line in f.readlines():
					if line.startswith("All"):
						d = line.split()
						pq, sq, rq = float(d[2]), float(d[3]), float(d[4])
					elif line.startswith("Things"):
						pq_thing = float(line.split()[2])
					elif line.startswith("Stuff"):
						pq_stuff = float(line.split()[2])
				f.close()
				return pq, pq_thing, pq_stuff, sq, rq

			if self.rank == 0:
				results, _ = temp
				out_dir = os.path.join(self.base_dir, 'test_model_log')
				if not os.path.exists(out_dir):
					os.mkdir(out_dir)
				out_file = '{}/test_model_result_{}'.format(out_dir, _list2str(model_cfg.tolist()[0]))
				if 'ade' in self.cfg.DATASETS.NAME.lower():
					print('Evaluating panoptic results on ADE...')
					os.system('sh bash_ade_evaluate.sh | tee {}'.format(out_file))
				elif 'coco' in self.cfg.DATASETS.NAME.lower():
					print('Evaluating panoptic results on COCO...')
					os.system('sh panoptic_scripts/bash_coco_nas_val_evaluate.sh {} | tee {}'.format(self.cfg.OUTPUT_DIR, out_file))
				else:
					raise NotImplementedError
				ap_box = round(results.results['bbox']['AP'] * 100, 2)
				ap_mask = round(results.results['segm']['AP'] * 100, 2)
				pq, pq_thing, pq_stuff, sq, rq = _parse(out_file)
				model_cfg.set(pq, pq_thing, pq_stuff, sq, rq, ap_box=ap_box, ap_mask=ap_mask)
				model_cfg.set_fitness(pq)
			synchronize() 
		return model_cfg

	def myprint(self, *s):
		if self.rank == 0:
			print(*s)

	def search(self, model, output_folders, dataset_names, distributed):
		self.myprint('[x]: Begin path priority search...')
		model.eval()
		if self.exist_cycle >= self.num_cycle: # cached results are enough
			self.myprint('[x]: Using cached model results')
			for i in range(self.num_cycle):
				models_this_cycle = self.cache_model_list[i*self.lcm : (i+1)*self.lcm]
				self.score_cycle_model(models_this_cycle)
		else:
			for i in range(self.exist_cycle, self.num_cycle):
				models_this_cycle = self.new_model_list[i*self.lcm : (i+1)*self.lcm]
				for i in range(len(models_this_cycle)):
					dataloaders_nas_val = make_data_loader(self.cfg, is_train=False, is_distributed=distributed)
					model_cfg = models_this_cycle[i]
					self.myprint('[x]: Evaluating model {}'.format(model_cfg))
					model_cfg = self._evaluate(model, model_cfg, dataloaders_nas_val, output_folders, dataset_names)
					self.myprint('[x]: Evaluation complete, saving checkpoint...')
					self.cache_model_list.append(model_cfg)
					self.myprint('[x]: Cached: ', self.cache_model_list)
					models_this_cycle[i] = model_cfg
					self.save_checkpoint(self.ckp)
				self.score_cycle_model(models_this_cycle)

		self.myprint('[x]: Searching complete, saving...')
		self.save_checkpoint("final_test_log.json")

