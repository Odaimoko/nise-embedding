#!/usr/bin/env bash
w=3
export CUDA_VISIBLE_DEVICES=1
~/anaconda3/bin/python batch_run_singleGPU.py --f 0 --t 12  --workers=$w  --nms_thres_1 $1 --nms_thres_2 $2&

export CUDA_VISIBLE_DEVICES=2
~/anaconda3/bin/python batch_run_singleGPU.py --f 12 --t 24  --workers=$w  --nms_thres_1 $1 --nms_thres_2 $2&

export CUDA_VISIBLE_DEVICES=3
~/anaconda3/bin/python batch_run_singleGPU.py --f 24 --t 36  --workers=$w  --nms_thres_1 $1 --nms_thres_2 $2&

export CUDA_VISIBLE_DEVICES=0
~/anaconda3/bin/python batch_run_singleGPU.py --f 36 --t 50 --workers=$w --nms_thres_1 $1 --nms_thres_2 $2


