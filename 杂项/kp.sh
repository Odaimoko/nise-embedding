#!/usr/bin/env bash
export PYTHONPATH=~/disk/posetrack/poseval/py-motmetrics:$PYTHONPATH
root_dir=pred_json-track
log_dir=bj_thres_out
files=$(ls $root_dir)
for f in $files
do
    echo $f
    ~/anaconda3/bin/python ~/disk/posetrack/poseval/py/evaluate.py
#    --groundTruth=pred_json-pre-commissioning/val_gt_task2/ --predictions=$root_dir/$f/ --evalPoseEstimation  --evalPoseTracking #  > $log_dir/$f.log &
done
