import os
import threading
import argparse


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
    os.system(cmd)


root_dir = get_args().root_dir
log_dir = os.path.join(root_dir, 'bj_thres_out')
mkdir(log_dir)

files = [f for f in os.listdir(root_dir) if 'valid' in f]
files = sorted(files)
ts = []
for f in files:
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
    print(cmd)
    os.system(cmd)
    # run(cmd)
    t = threading.Thread(target = run, args = (cmd,))
    # t.start()
    ts.append(t)
for t in ts:
    t.start()
for t in ts:
    t.join()
