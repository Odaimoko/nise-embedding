#
# ──────────────────────────────────────────────────────────────────────────────────── I ──────────
#   :::::: U S E   P R E T R A I N E D   M O D E L S   T O   T E S T
#       R E - I M P L E M E N T E D  S I M P L E   B A S E L I N E :
# ────────────────────────────────────────────────────────────────────────────

from pathlib import PurePosixPath
import argparse
import pprint
import os
import torch.multiprocessing as mp

from multiprocessing.pool import Pool
from multiprocessing import Lock

# local packages
import _init_paths
from nise_lib.nise_config import nise_cfg, nise_logger
from nise_lib.nise_functions import *
from nise_lib.nise_debugging_func import *
from nise_lib.core import *
from tron_lib.core.config import cfg as tron_cfg
from nise_lib.frameitem import FrameItem

from simple_lib.core.config import config as simple_cfg

write_lock = Lock()


def one_img(img, img_dir, det_result, maskRCNN):
    img_path = os.path.join(img_dir, img)
    print(img_path)
    fi = FrameItem(nise_cfg, simple_cfg, img_path, 1, True, None)
    fi.detect_human(maskRCNN)
    write_lock.acquire()
    det_result.update({img: fi.detect_results()})
    write_lock.release()


if __name__ == '__main__':
    mp.set_start_method('spawn', force = True)
    
    human_detect_args = human_detect_parse_args()
    maskRCNN, human_det_dataset = load_human_detect_model(human_detect_args, tron_cfg)
    
    # img_dir = 'data/coco/images/test2017'
    
    dirs = ['val']
    for dir in dirs:
        img_dir = 'data/coco/images/' + dir + '2017'
        images = os.listdir(img_dir)
        print(len(images))
        det_result = {}
        # all_ims = [(img, img_dir, det_result, maskRCNN) for img in images]
        # with Pool(processes = 5) as p:
        #     p.starmap(one_img, all_ims, chunksize = 1000)
        for img in images:
            if not '00885.jpg' in img:
                continue
            img_path = os.path.join(img_dir, img)
            print(img_path)
            fi = FrameItem(nise_cfg, simple_cfg, img_path, 1, True, None)
            fi.detect_human(maskRCNN)
            det_result.update({img: fi.detect_results()})

        # json_path = 'run_coco_' + dir + '_det.json'
        # with open(json_path, 'w') as f:
        #     json.dump(det_result, f)
        #     debug_print('coco json saved:', json_path)
