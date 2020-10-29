ITER=600000
BS=512
LR=0.25

cd $CURRENT_DIR"/finetune-ImageNet"

python train.py --model-size 3.8G --total-iters $ITER --batch-size $BS --learning-rate $LR
