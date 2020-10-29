# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
import json

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug

def build_id_dict(annotation, is_coco):
    """
    label_on_pic (continuous) -> true category id (discrete)
    Param annotation: 'categories'
    """
    # THING_LABEL = 0

    c2d = dict()
    if is_coco: # 1 thing class
        cur = 0
        for ann in annotation:
            if ann['isthing'] == 1:
                continue
            cur += 1 # start from 1, 0 for thing
            c2d[cur] = ann['id']
        return c2d
    else: # thing class seperate
        cur = 0
        for ann in annotation:
            c2d[cur] = ann['id']
            cur += 1
        return c2d

def compute_on_dataset(model, data_loader, device, bbox_aug, c2d_json_path, timer=None, rngs=None, test_only=False):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    ann_json = json.load(open(c2d_json_path, 'r'))
    seg_result = []
    is_coco = ('coco' in c2d_json_path or 'ade' in c2d_json_path)
    c2d = build_id_dict(ann_json['categories'], is_coco=is_coco)

    for _, batch in enumerate(tqdm(data_loader)):
        if test_only:
            images, _, _, image_ids, img_ids, ori_sizes = batch
        else:
            images, targets, _, image_ids, img_ids, ori_sizes = batch
        images = images.to(device)
        with torch.no_grad():
            if timer:
                timer.tic()
            if bbox_aug:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images, img_ids=img_ids, c2d=c2d, seg_result=seg_result, ori_sizes=ori_sizes, rngs=rngs)
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict, seg_result


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, seg_result):
    all_predictions = all_gather(predictions_per_gpu)
    all_seg = all_gather(seg_result)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    seg_pred = []
    for s in all_seg:
        seg_pred.extend(s)
    seg_json_fn = 'semantic_seg_pred.json'
    with open(seg_json_fn, 'w') as f:
        json.dump(seg_pred, f)
    print('Saved segmentation json file to {}'.format(seg_json_fn))
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions

def bn_statistic(model, rngs, cfg):
    from maskrcnn_benchmark.data import make_data_loader
    device = cfg.MODEL.DEVICE
    import torch.nn as nn

    for name, param in model.named_buffers():
        if 'running_mean' in name:
            nn.init.constant_(param, 0)
        if 'running_var' in name:
            nn.init.constant_(param, 1)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=True
    )
    model.train()
    pbar = tqdm(total=500)
    for iteration, (images, targets, segment_target, _, img_ids, ori_sizes) in enumerate(data_loader, 1):
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        with torch.no_grad():
            loss_dict = model(images, targets, segment_target, img_ids=img_ids, c2d=None, ori_sizes=ori_sizes, rngs=rngs)
        pbar.update(1)
        if iteration >= 500:
            break
    pbar.close()
    return model

def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        bbox_aug=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        c2d_json_path=None,
        rngs=None,
        cfg=None,
        test_only=None,
):
    # import resource
    # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    # calibrate bn statistics
    if 'search' in cfg.MODEL.BACKBONE.CONV_BODY:
        print('recalibrate bn')
        model = bn_statistic(model, rngs, cfg)
        print('recalibrate finished!')
        model.eval()
    predictions, seg_result = compute_on_dataset(model, data_loader, device, bbox_aug=bbox_aug, c2d_json_path=c2d_json_path, rngs=rngs, test_only=test_only)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions, seg_result)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        # test_only=test_only
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
