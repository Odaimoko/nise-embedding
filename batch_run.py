import copy
import os
from multiprocessing.pool import Pool

import yaml


def mkdir(path):
    path = path.strip().rstrip("\\")
    
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False


series = [(2 * i, 2 * i + 2) for i in range(3,25)]
print(series)

def run_cmd(cmd):
    print('Running:', cmd)
    os.system(cmd)
    # time.sleep(2)


def create_yaml(series):
    with open('exp_config/t.yaml', 'r')as f:
        c = yaml.load(f)
    out_dir = 'exp_config/batch_%02d_%02d' % (series[0][0], series[-1][-1])
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


batch_files = create_yaml(series)

m = '''~/anaconda3/bin/python run.py --model FlowNet2S --flownet_resume ../flownet2-pytorch/FlowNet2-S_checkpoint.pth.tar --simple_cfg ../simple-baseline-pytorch/experiments/pt17/res50-coco-256x192_d256x3_adam_lr1e-3.yaml --gpus 0 --simple-model-file /home/zhangxt/disk/posetrack/simple-baseline-pytorch/output-pt17/pt17/pose_resnet_50/res50-coco-256x192_d256x3_adam_lr1e-3/pt17-epoch-20-87.92076779477024 --tron_cfg /home/zhangxt/disk/posetrack/nise_embedding/my_e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml --load_detectron /home/zhangxt/disk/pretrained/e2e_mask_rcnn_X-101-64x4d-FPN_1x.pkl --dataset coco2017 --nise_config '''

# os.environ['CUDA_VISIBLE_DEVIES'] = '3'
# cuda = 'export CUDA_VISIBLE_DEVICES=3'
# run_cmd(cuda) # this is a must
# print(os.environ['CUDA_VISIBLE_DEVIES'])
p = Pool(4)
p.map(run_cmd, [m + f for f in batch_files])
# run_cmd(m+'exp_config/t.yaml')
