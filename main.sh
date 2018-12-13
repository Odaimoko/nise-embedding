#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
~/anaconda3/bin/python batch_run_singleGPU.py --f 0 --t 2 &
export CUDA_VISIBLE_DEVICES=2
~/anaconda3/bin/python batch_run_singleGPU.py --f 2 --t 4 &
export CUDA_VISIBLE_DEVICES=3
~/anaconda3/bin/python batch_run_singleGPU.py --f 4 --t 6
