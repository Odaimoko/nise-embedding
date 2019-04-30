import os
import threading
import argparse
from multiprocessing import Pool


def mkdir(path):
    path = path.strip().rstrip("\\")
    
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False


def get_args():
    p = argparse.ArgumentParser(description = 'Evaluate all folders under ROOT')
    
    p.add_argument('--root', dest = 'root_dir', required = True)
    args = p.parse_args()
    return args


def run(cmd):
    print(cmd)
    os.system(cmd)
    print('DONEDONEDONE...!')


root_dir = get_args().root_dir
log_dir = os.path.join(root_dir, 'bj_thres_out')
mkdir(log_dir)

files = [f for f in os.listdir(root_dir) if 'valid' in f]
files = sorted(files)
ts = []
for f in files:
    if ('joint_0.3' in f or 'joint_0.4' in f or 'joint_0.5' in f or 'joint_0.6' in f or 'joint_0.7' in f) \
            and ('box_0.4' in f or 'box_0.5' in f or 'box_0.6' in f or 'box_0.7' in f or 'box_0.8' in f):
        print(f)
        cmd = ' '.join([
            'python ',
            '../poseval/py/evaluate.py',
            '--groundTruth=pred_json-pre-commissioning/val_gt_task1/ ',
            '--predictions=' + root_dir + '/' + f + '/',
            ' --evalPoseEstimation  --evalPoseTracking ',
            ">",
            log_dir + "/" + f + ".log",
        ])
        ts.append((cmd,))

with Pool(processes = 12) as po:
    print('Pool created.')
    po.starmap(run, ts)
