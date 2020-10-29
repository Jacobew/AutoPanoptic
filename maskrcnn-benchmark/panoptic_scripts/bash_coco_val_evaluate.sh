python panopticapi/combine.py \
	--semseg_json_file semantic_seg_pred.json \
	--instseg_json_file $1/inference/coco_2017_val/segm.json \
	--images_json_file datasets/coco/annotations/panoptic_val2017.json \
	--categories_json_file datasets/coco/annotations/panoptic_coco_categories.json \
	--panoptic_json_file panoptic_coco2017_val.json
python panopticapi/evaluation.py \
	--gt_json_file datasets/coco/annotations/panoptic_val2017.json \
	--gt_folder datasets/coco/annotations/panoptic_val2017 \
	--pred_json_file panoptic_coco2017_val.json
