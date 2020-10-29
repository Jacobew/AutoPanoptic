# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor
import json

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import (MetricLogger, TensorboardLogger)
from maskrcnn_benchmark.modeling.search_space import head_ss_keys, inter_ss_keys
from maskrcnn_benchmark.modeling.backbone.search_space import blocks_key
from maskrcnn_benchmark.engine.architecture_search import PathPrioritySearch
from maskrcnn_benchmark.utils.timer import Timer, get_time_str

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def train(cfg, local_rank, distributed, model_config=None, use_tensorboard=True):
    model = build_detection_model(cfg, model_config)
    if get_rank() == 0:
        if 'search' in cfg.MODEL.BACKBONE.CONV_BODY:
            print('backbone search space:', blocks_key)
        else:
            print('backbone:', cfg.MODEL.BACKBONE)
        if 'search' in cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR or 'search' in cfg.MODEL.SEG_BRANCH.SEGMENT_BRANCH:
            print('head search space:', head_ss_keys)
        else:
            print('head:', cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR, cfg.MODEL.SEG_BRANCH.SEGMENT_BRANCH)
        if 'search' in cfg.MODEL.INTER_MODULE.NAME:
            print('inter search space:', inter_ss_keys)
        else:
            print('inter:', cfg.MODEL.INTER_MODULE.NAME)
        print(model)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer, lr_dict = make_optimizer(cfg, model)
    if get_rank() == 0:
        for item in lr_dict:
            print(item)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    if not ('search' in cfg.MODEL.BACKBONE.CONV_BODY or
         'search' in cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR or
         'search' in cfg.MODEL.SEG_BRANCH.SEGMENT_BRANCH):
        use_mixed_precision = cfg.DTYPE == "float16"
        amp_opt_level = 'O1' if use_mixed_precision else 'O0'
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True 
        )

    # if 'search' in cfg.MODEL.BACKBONE.CONV_BODY:
    #     def forward_hook(module: Module, inp: (Tensor,)):
    #         if module.weight is not None:
    #             module.weight.requires_grad = True
    #         if module.bias is not None:
    #             module.bias.requires_grad = True

    #     all_modules = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.GroupNorm, ) # group norm更新！！
    #     for m in model.modules():
    #         if isinstance(m, all_modules):
    #             m.register_forward_pre_hook(forward_hook)

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_for_period=True)
    else:
        data_loader_val = None

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if use_tensorboard:
        meters = TensorboardLogger(
            cfg=cfg,
            log_dir=cfg.TENSORBOARD_EXPERIMENT,
            start_iter=arguments['iteration'],
            delimiter="  ")
    else:
        meters = MetricLogger(delimiter="  ")

    do_train(
        cfg,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        test_period,
        arguments,
        meters,
    )

    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    # output_folders = [None] * len(cfg.DATASETS.TEST)
    # dataset_names = cfg.DATASETS.TEST
    dataset_names = cfg.DATASETS.NAS_VAL if not cfg.NAS.TRAIN_SINGLE_MODEL else cfg.DATASETS.TEST
    output_folders = [None] * len(dataset_names)

    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)

    if cfg.NAS.TRAIN_SINGLE_MODEL:
        if get_rank() == 0:
            print('==' * 20 , 'Evaluating single model...', '=='*20)
        for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
            inference(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                bbox_aug=cfg.TEST.BBOX_AUG.ENABLED,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
                c2d_json_path=cfg.MODEL.SEG_BRANCH.JSON_PATH,
                cfg=cfg,
            )
            synchronize()
        if get_rank() == 0:
            if 'coco' in cfg.DATASETS.NAME.lower():
                print('Evaluating panoptic results on COCO...')
                os.system('sh panoptic_scripts/bash_coco_val_evaluate.sh {} | tee pq_results'.format(cfg.OUTPUT_DIR))
    elif not cfg.NAS.SKIP_NAS_TEST:
        if get_rank() == 0:
            print('==' * 10 , 'Start NAS testing', '=='*10)
        timer = Timer()
        timer.tic()
        searcher = PathPrioritySearch(cfg, base_dir='./nas_test')
        searcher.generate_fair_test() # load cache results and generate new model for test
        searcher.search(model, output_folders, dataset_names, distributed)
        searcher.save_topk()
        total_time = timer.toc()
        total_time_str = get_time_str(total_time)
        if get_rank() == 0:
            print('Finish NAS testing, total time:{}'.format(total_time_str))
        os._exit(0)
    else:
        print('Skipping NAS testing...')


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        "--build-model",
        default="",
        metavar="FILE",
        help="path to NAS model build file",
        type=str,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    if cfg.NAS.TRAIN_SINGLE_MODEL:
        assert len(args.build_model) != 0, 'args.build_model should be provided'
        model_config = json.load(open(args.build_model, 'r'))
        if isinstance(model_config, list):
            assert len(model_config) == 1
            model_config = model_config[0]
        print('Training single model:', model_config)
        model = train(cfg, args.local_rank, args.distributed, model_config)
    else:
        model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
