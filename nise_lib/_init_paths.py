import sys
import os.path as osp

paths = [
    '/home/zhangxt/disk/posetrack/flownet2-pytorch/',
    '/home/zhangxt/disk/posetrack/flownet2-pytorch/flow_lib',
    '/home/zhangxt/disk/posetrack/simple-baseline-pytorch/',
    '/home/zhangxt/disk/posetrack/simple-baseline-pytorch/simple_lib',
    '/home/zhangxt/disk/posetrack/Detectron.pytorch/',
    '/home/zhangxt/disk/posetrack/Detectron.pytorch/tron_lib',
]


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        print('add', path)


this_dir = osp.dirname(__file__)

for p in paths:
    # lib_path = osp.join(this_dir, '..', p)
    lib_path = p
    add_path(lib_path)
