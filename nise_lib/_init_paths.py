import sys
import os.path as osp

paths = [
    '../flownet2-pytorch/',
    '../simple-baseline-pytorch/',
    '../Detectron.pytorch/',
]


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

for p in paths:
    lib_path = osp.join(this_dir, '..', p)
    print('add',lib_path)
    add_path(lib_path)
