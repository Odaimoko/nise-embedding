#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
~/anaconda3/bin/python batch_run_singleGPU.py --f 0 --t 16 &
export CUDA_VISIBLE_DEVICES=2
~/anaconda3/bin/python batch_run_singleGPU.py --f 16 --t 32 &
export CUDA_VISIBLE_DEVICES=3
~/anaconda3/bin/python batch_run_singleGPU.py --f 32 --t 50
#&
#export CUDA_VISIBLE_DEVICES=0
#~/anaconda3/bin/python batch_run_singleGPU.py --f 10 --t 11
