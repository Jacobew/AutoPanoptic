# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import json


import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.modeling.search_space import head_ss_keys, inter_ss_keys
from maskrcnn_benchmark.modeling.backbone.search_space import blocks_key
from maskrcnn_benchmark.engine.architecture_search import PathPrioritySearch
from maskrcnn_benchmark.utils.timer import Timer, get_time_str

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
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
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    assert len(args.build_model) != 0, 'args.build_model should be provided'
    model_config = json.load(open(args.build_model, 'r'))
    if isinstance(model_config, list):
        assert len(model_config) == 1
        model_config = model_config[0]
    print('Testing single model:', model_config)
    

    model = build_detection_model(cfg, model_config)
    model.to(cfg.MODEL.DEVICE)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

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
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, test_only=cfg.TEST_ONLY)

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
                test_only=cfg.TEST_ONLY
            )
            synchronize()
    elif not cfg.NAS.SKIP_NAS_TEST:
        if get_rank() == 0:
            print('==' * 10 , 'Start NAS testing', '=='*10)
        timer = Timer()
        timer.tic()
        searcher = PathPrioritySearch(cfg, './nas_test')
        searcher.generate_fair_test() # load cache results and generate new model for test
        searcher.search(model, output_folders, dataset_names, distributed)
        searcher.save_topk()
        total_time = timer.toc()
        total_time_str = get_time_str(total_time)
        if get_rank() == 0:
            print('Finish NAS testing, total time:{}'.format(total_time_str))
        return
    else:
        print('Skipping NAS testing...')


if __name__ == "__main__":
    main()
