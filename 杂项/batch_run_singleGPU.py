import copy
import os
from multiprocessing.pool import Pool
import argparse
import yaml
import time
import threading
import random


def get_nise_arg_parser():
    parser = argparse.ArgumentParser(description = 'NISE PT')
    parser.add_argument('--workers', default = 4, type = int, metavar = 'N',
                        help = 'number of data loading workers (default: 12)')
    parser.add_argument('--num_gpus', default = 1, type = int, metavar = 'N',
                        help = 'number of GPU to use (default: 1)')
    
    parser.add_argument('--f', type = int)
    parser.add_argument('--t', type = int, )
    parser.add_argument('--nms_thres_1', type = float, default = .05)
    parser.add_argument('--nms_thres_2', type = float, default = .5)
    
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
    with open('exp_config/t-flow-debug.yaml', 'r')as f:
        c = yaml.load(f)
    training_start_time = time.strftime("%m_%d-%H_%M", time.localtime())
    
    out_dir = 'exp_config/%s-batch' % (training_start_time,)  # series[0][0], series[-1][-1])
    mkdir(out_dir)
    batch_files = []
    for s in series:
        nc = copy.deepcopy(c)
        nc['TEST']['FROM'] = s[0]
        nc['TEST']['TO'] = s[1]
        nc['ALG']['UNIFY_NMS_THRES_1'] = a.nms_thres_1
        nc['ALG']['UNIFY_NMS_THRES_2'] = a.nms_thres_2
        file_name = 'batch_%02d_%02d-nmsthres-%.2f,%.2f.yaml' % (s[0], s[1], a.nms_thres_1, a.nms_thres_2)
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
print(series)

batch_files = create_yaml(series)

m = '''~/anaconda3/bin/python run.py --model FlowNet2S --flownet_resume ../flownet2-pytorch/FlowNet2-S_checkpoint.pth.tar --simple_cfg ../simple-baseline-pytorch/experiments/pt17/res50-coco-256x192_d256x3_adam_lr1e-3.yaml --gpus 0 --simple-model-file /home/zhangxt/disk/posetrack/simple-baseline-pytorch/output-pt17/pt17/pose_resnet_50/res50-coco-256x192_d256x3_adam_lr1e-3/pt17-epoch-20-87.92076779477024 --tron_cfg /home/zhangxt/disk/posetrack/nise_embedding/my_e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml --load_detectron /home/zhangxt/disk/pretrained/e2e_mask_rcnn_X-101-64x4d-FPN_1x.pkl --dataset coco2017 --nise_config '''

p = Pool(a.workers)
p.map(run_cmd, [m + f for f in batch_files])
p.close()
p.join()
# threads = []
# for f in batch_files:
#     threads.append(threading.Thread(target = run_cmd, args = [f]))
# for t in threads:
#     t.start()
# for t in threads:
#     t.join()
print(a)
print('DONEDONEDONE')
