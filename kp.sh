#!/usr/bin/env bash
export PYTHONPATH=~/disk/posetrack/poseval/py-motmetrics:$PYTHONPATH

files=$(ls pred_json)
for f in $files
do
    echo $f
    ~/anaconda3/bin/python ~/disk/posetrack/poseval/py/evaluate.py --groundTruth=pred_json-pre-commissioning/val_gt_task2/ --predictions=pred_json/$f/ --evalPoseEstimation
done
