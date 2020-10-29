#parameters
EXP="eval_autopanoptic_on_coco"
STORE_DIR="./cachemodel/"
NAME="coco"
OUTPUT_DIR=${STORE_DIR}${EXP}
INFO="eval_autopanoptic_on_coco"
config="configs/autopanoptic/autopanoptic_1x.yaml"
NGPUS=8
TEST_BATCH_SIZE=16
BUILD="./test_models/COCO_3.8G"
TRAIN_SINGLE_MODEL=True
TRAIN_SUPERNET=False
BACKBONE="AutoPanoptic-COCO-FPN"
BOX="FPN2MLPFeatureExtractor"
MASK="AutoPanoptic_MaskRCNNFPNFeatureExtractor"
SEG="AutoPanoptic_Segmentation_Branch"
WEIGHT=$CURRENT_DIR/pretrain/autopanoptic.pth
STEM_OUT_CHANNELS=72
WORKERS=4

cd $CURRENT_DIR/maskrcnn-benchmark

python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file $config --build-model $BUILD \
TEST.IMS_PER_BATCH $TEST_BATCH_SIZE TEST.IMS_PER_BATCH $TEST_BATCH_SIZE INFO $INFO DATALOADER.NUM_WORKERS $WORKERS \
MODEL.BACKBONE.CONV_BODY $BACKBONE MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR $BOX MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR $MASK \
MODEL.SEG_BRANCH.SEGMENT_BRANCH $SEG MODEL.RESNETS.STEM_OUT_CHANNELS $STEM_OUT_CHANNELS \
NAS.TRAIN_SUPERNET $TRAIN_SUPERNET NAS.TRAIN_SINGLE_MODEL $TRAIN_SINGLE_MODEL \
OUTPUT_DIR $OUTPUT_DIR DATASETS.NAME $NAME \
MODEL.WEIGHT $WEIGHT \


cd $CURRENT_DIR/maskrcnn-benchmark

python panopticapi/combine.py \
	--semseg_json_file semantic_seg_pred.json \
	--instseg_json_file $OUTPUT_DIR/inference/coco_2017_val/segm.json \
	--images_json_file datasets/coco/annotations/panoptic_val2017.json \
	--categories_json_file datasets/coco/annotations/panoptic_coco_categories.json \
	--panoptic_json_file panoptic_coco2017_val.json | tee combine_log
python panopticapi/evaluation.py \
	--gt_json_file datasets/coco/annotations/panoptic_val2017.json \
	--gt_folder datasets/coco/annotations/panoptic_val2017 \
	--pred_json_file panoptic_coco2017_val.json | tee eval_log

