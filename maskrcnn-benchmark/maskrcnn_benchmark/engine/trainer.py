# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time
import math

import torch
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.comm import get_world_size, synchronize
# from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.backbone.search_space import blocks_key
from maskrcnn_benchmark.modeling.search_space import head_ss_keys, inter_ss_keys
from apex import amp


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def broadcast_data(data):
    if not torch.distributed.is_initialized():
        return data
    rank = dist.get_rank()
    if rank == 0:
        data_tensor = torch.tensor(data + [0], device="cuda")
    else:
        data_tensor = torch.tensor(data + [1], device="cuda")
    torch.distributed.broadcast(data_tensor, 0)
    while data_tensor.cpu().numpy()[-1] == 1:
        time.sleep(1)

    return data_tensor.cpu().numpy().tolist()[:-1]


def generate_rng(layers, ss_size, lcm):
    rngs = np.ones([layers, lcm], dtype=np.int8)
    for i in range(layers):
        rng_per_layer = list()
        for j in range(lcm//ss_size):
            rng_per_layer.extend(np.random.permutation(ss_size).tolist())
        rngs[i] = rng_per_layer
    return rngs

def do_train(
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
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    # meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    # dataset_names = cfg.DATASETS.TEST

    backbone_rngs, head_rngs, inter_rngs, rng, rngs = None, None, None, None, None

    if 'search' in cfg.MODEL.BACKBONE.CONV_BODY or \
        'search' in cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR or \
        'search' in cfg.MODEL.SEG_BRANCH.SEGMENT_BRANCH:
        # synchronize rngs

        num_states = sum(cfg.MODEL.BACKBONE.STAGE_REPEATS)
        if cfg.MODEL.SEG_BRANCH.SHARE_SUBNET:
            head_layers = len(cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS) + cfg.MODEL.SEG_BRANCH.SUBNET_DEPTH
        else:
            head_layers = len(cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS) + 4 * cfg.MODEL.SEG_BRANCH.SUBNET_DEPTH
        inter_layers = cfg.NAS.INTER_LAYERS
        backbone_ss_size = len(blocks_key)
        head_ss_size = len(head_ss_keys)
        inter_ss_size = cfg.NAS.INTER_SIZE
        
        if 'search' in cfg.MODEL.BACKBONE.CONV_BODY:
            _lcm = backbone_ss_size*head_ss_size//math.gcd(backbone_ss_size, head_ss_size)
            lcm = inter_ss_size*_lcm//math.gcd(inter_ss_size, _lcm)
        else:
            lcm = inter_ss_size*head_ss_size//math.gcd(inter_ss_size, head_ss_size)

        # print('lcm:', lcm)
    fwd_idx = -1
    for iteration, (images, targets, segment_target, _, img_ids, ori_sizes) in enumerate(data_loader, start_iter):
        
        if any(len(target) < 1 for target in targets):
            logger.error("Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        if 'search' in cfg.MODEL.BACKBONE.CONV_BODY or \
        'search' in cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR or \
        'search' in cfg.MODEL.SEG_BRANCH.SEGMENT_BRANCH:
            if rngs is None or iteration % lcm == 0:
                del rngs
                if 'search' in cfg.MODEL.BACKBONE.CONV_BODY:
                    backbone_rngs = generate_rng(num_states, backbone_ss_size, lcm)
                head_rngs = generate_rng(head_layers, head_ss_size, lcm)
                inter_rngs = generate_rng(inter_layers, inter_ss_size, lcm)
                if 'search' in cfg.MODEL.BACKBONE.CONV_BODY:
                    rngs = np.concatenate([backbone_rngs, head_rngs, inter_rngs], axis=0).transpose(1, 0)
                    del backbone_rngs
                else:
                    rngs = np.concatenate([head_rngs, inter_rngs], axis=0).transpose(1, 0)
                del head_rngs, inter_rngs

            rng = rngs[iteration % lcm]
            rng = broadcast_data(rng.tolist())

            fwd_idx = (fwd_idx + 1) % lcm

            loss_dict = model(images, targets, segment_target, img_ids=img_ids, c2d=None, ori_sizes=ori_sizes, rngs=rng)
            del rng
        else:
            loss_dict = model(images, targets, segment_target, img_ids=img_ids, c2d=None, ori_sizes=ori_sizes)

        if cfg.MODEL.SEG_BRANCH.ADD_SEG_BRANCH:
            segmentation_loss = loss_dict.pop("loss_segmentation")
            losses = cfg.MODEL.SEG_BRANCH.LAMDA_INSTANCE * sum(loss for loss in loss_dict.values()) + cfg.MODEL.SEG_BRANCH.LAMDA_SEGMENTATION * segmentation_loss
            loss_dict['loss_segmentation'] = segmentation_loss # reproduce the complete loss
        else:
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        meters.update(loss=losses.item(), **loss_dict_reduced)

        if 'search' in cfg.MODEL.BACKBONE.CONV_BODY or \
            'search' in cfg.MODEL.SEG_BRANCH.SEGMENT_BRANCH or \
            'search' in cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR:
            if fwd_idx == 0:
                optimizer.zero_grad()
            losses.backward()
            if fwd_idx == lcm - 1:
                optimizer.step()
        else:
            optimizer.zero_grad()
            # losses.backward()
            # Note: If mixed precision is not used, this ends up doing nothing
            # Otherwise apply loss scaling for mixed-precision recipe
            with amp.scale_loss(losses, optimizer) as scaled_losses:
                scaled_losses.backward()
            optimizer.step()

        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        del loss_dict, losses, images, targets
        # torch.cuda.empty_cache()

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if data_loader_val is not None and test_period > 0 and iteration % test_period == 0:
            meters_val = MetricLogger(delimiter="  ")
            synchronize()
            _ = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                model,
                # The method changes the segmentation mask format in a data loader,
                # so every time a new data loader is created:
                make_data_loader(cfg, is_train=False, is_distributed=(get_world_size() > 1), is_for_period=True),
                dataset_name="[Validation]",
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=None,
                c2d_json_path=cfg.MODEL.SEG_BRANCH.JSON_PATH,
                cfg=cfg,
            )
            synchronize()
            model.train()
            with torch.no_grad():
                # Should be one image for each GPU:
                for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                    images_val = images_val.to(device)
                    targets_val = [target.to(device) for target in targets_val]
                    loss_dict = model(images_val, targets_val)
                    # losses = sum(loss for loss in loss_dict.values())
                    if cfg.MODEL.SEG_BRANCH.ADD_SEG_BRANCH:
                        segmentation_loss = loss_dict.pop("loss_segmentation")
                        losses = cfg.MODEL.SEG_BRANCH.LAMDA_INSTANCE * sum(loss for loss in loss_dict.values()) + cfg.MODEL.SEG_BRANCH.LAMDA_SEGMENTATION * segmentation_loss
                        loss_dict['loss_segmentation'] = segmentation_loss # reproduce the complete loss
                    else:
                        losses = sum(loss for loss in loss_dict.values())
                    loss_dict_reduced = reduce_loss_dict(loss_dict)
                    meters_val.update(loss=losses.item(), **loss_dict_reduced)
            synchronize()
            logger.info(
                meters_val.delimiter.join(
                    [
                        "[Validation]: ",
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters_val),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
