
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path as osp

paths = [
    '.',
    'other_libs/',
    'other_libs/flownet_utils/',
    'other_libs/simple_lib/',
    'other_libs/tron_lib/',
    'other_libs/hr_lib/',
]


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        print('add', path)


this_dir = osp.dirname(__file__)

for p in paths:
    lib_path = osp.join(this_dir, '..', p)
    # lib_path = p
    add_path(lib_path)
