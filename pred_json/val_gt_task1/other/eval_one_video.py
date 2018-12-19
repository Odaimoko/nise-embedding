
import os
import sys
import re
import pathlib
from subprocess import *

def run_cmd(cmd_list):
    cmd = " ".join(cmd_list)
    print('Running:', cmd)
    # os.system(cmd)


gt_jsons = os.listdir('other')
gt_jsons = sorted(gt_jsons)
for js in gt_jsons:  # no DS_Store
    if not '.json' in js:
        continue
    # print(js)
    cmd = ['mv', os.path.join('other', js), '.']
    run_cmd(cmd)
    eval_cmd = ["bash", '/Users/oda/posetrack/nise_embedding/pred_json/val_gt_task1/eval_task_1.sh']
    run_cmd(eval_cmd)
    mv_back_cmd = ['mv', js, 'other']
    run_cmd(mv_back_cmd)
