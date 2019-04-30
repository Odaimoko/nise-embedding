import copy
import os
from multiprocessing.pool import Pool
import argparse
import yaml
import time
import threading
import random
from pprint import pprint


def get_nise_arg_parser():
    parser = argparse.ArgumentParser(description = 'NISE PT')
    parser.add_argument('--w', default = 2, type = int, metavar = 'N',
                        help = 'number of data loading workers (default: 12)')
    parser.add_argument('--g', default = 1, type = int, metavar = 'N',
                        help = 'number of GPU to use (default: 1)')
    
    parser.add_argument('--f', type = int, default = 0)
    parser.add_argument('--t', type = int, default = 50)
    
    args, rest = parser.parse_known_args()
    return args


def mkdir(path):
    path = path.strip().rstrip("\\")
    
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False


def create_yaml(series):
    with open('exp_config/3/t-3-root-mnet-gen-all-distmat.yaml', 'r')as f:
        c = yaml.load(f)
    training_start_time = time.strftime("%m_%d-%H_%M", time.localtime())
    
    out_dir = 'exp_config/batch/%s-batch/' % (training_start_time,)  # series[0][0], series[-1][-1])
    mkdir(out_dir)
    batch_files = []
    for s in series:
        nc = copy.deepcopy(c)
        nc['TEST']['FROM'] = s[0]
        nc['TEST']['TO'] = s[1]
        file_name = 'batch_%02d_%02d.yaml' % (s[0], s[1])
        long_file_name = os.path.join(out_dir, file_name)
        batch_files.append(long_file_name)
        with open(long_file_name, 'w')as f:
            yaml.dump(nc, f)
    return batch_files


def run_cmd(cmd):
    print('Running:', cmd)
    os.system(cmd)
    time.sleep(random.randint(3, 10))


a = get_nise_arg_parser()
v = 1
series = [(v * i, v * i + v) for i in range(int(a.f / v), int(a.t / v))]
if not series:
    print('ERR: TO must be larger than FROM.')
    exit(1)
pprint(series)

batch_files = create_yaml(series)
pprint(batch_files)
m = '''python scripts/run-mnet.py  --tron_cfg exp_config/detectron/my_e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml --load_detectron ~/zhangxt/disk/pretrained/e2e_mask_rcnn_X-101-64x4d-FPN_1x.pkl  --dataset coco2017 --task1pred pred_json-single-est/79.0-sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50 --nise_config '''

p = Pool(a.w)
p.map(run_cmd, [m + f for f in batch_files])
p.close()
p.join()


# threads = []
# for f in batch_files:
#     threads.append(threading.Thread(target = run_cmd, args = [f]))
#
# for t in threads:
#     t.start()
# for t in threads:
#     t.join()
print(a)
print('DONEDONEDONE')
