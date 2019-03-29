cd /Users/oda/posetrack/poseval/py 
export PYTHONPATH=/Users/oda/posetrack/poseval/py-motmetrics:$PYTHONPATH
/usr/local/bin/python2.7 /Users/oda/posetrack/poseval/py/evaluate.py --groundTruth=/Users/oda/posetrack/nise_embedding/pred_json/val_gt_task1/ --predictions=/Users/oda/posetrack/nise_embedding/pred_json/valid_task_1_DETbox_propfiltered_propthres_1_propDET/ --evalPoseEstimation