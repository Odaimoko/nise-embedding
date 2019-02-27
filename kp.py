import os
import threading

def run(cmd):
    os.system(cmd)
# export PYTHONPATH=~/disk/posetrack/poseval/py-motmetrics:$PYTHONPATH
root_dir = 'pred_json-track'
log_dir = 'bj_thres_out'
files = os.listdir(root_dir)
files = sorted(files)
ts = []
for f in files:
    print(f)
    cmd = ' '.join([
        '~/anaconda3/bin/python ',
        '~/disk/posetrack/poseval/py/evaluate.py ',
        '--groundTruth=pred_json-pre-commissioning/val_gt_task2/ ',
        '--predictions=' + root_dir + '/' + f+'/',
        ' --evalPoseEstimation  --evalPoseTracking ',
        ">",
        log_dir + "/" + f + ".log",
    
    ])
    # print(cmd)
    # os.system(cmd)
    t = threading.Thread(target = run, args = (cmd,))
    t.start()
    ts.append(t)
for t in ts:
    t.join()
