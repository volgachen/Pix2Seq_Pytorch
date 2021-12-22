DATA_PATH=./coco
NGPUS=$1
BATCH_SIZE=16

python -m torch.distributed.launch --nproc_per_node $NGPUS --master_port 12211 --use_env main.py \
 --coco_path $DATA_PATH --pix2seq_lr --large_scale_jitter --rand_target --model pix2seq --pre_norm\
 --backbone resnet101 --lr 3e-3 --lr_backbone 3e-3 --batch_size $BATCH_SIZE\
 --max_input_size 640 --max_objects 100 --num_bins 500 --epochs 200 ${@:2}