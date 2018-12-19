#!/usr/bin/env bash
w=3
export CUDA_VISIBLE_DEVICES=1
~/anaconda3/bin/python batch_run_singleGPU.py --f 0 --t 2  --workers=$w  --nms_thres_1 $1 --nms_thres_2 $2&

export CUDA_VISIBLE_DEVICES=2
~/anaconda3/bin/python batch_run_singleGPU.py --f 2 --t 4  --workers=$w  --nms_thres_1 $1 --nms_thres_2 $2&

#export CUDA_VISIBLE_DEVICES=3
#~/anaconda3/bin/python batch_run_singleGPU.py --f 4 --t 6  --workers=$w  --nms_thres_1 $1 --nms_thres_2 $2&

export CUDA_VISIBLE_DEVICES=0
~/anaconda3/bin/python batch_run_singleGPU.py --f 4 --t 6 --workers=$w --nms_thres_1 $1 --nms_thres_2 $2


