import os
import os.path
import sys
import numpy as np
import  argparse

def get_arg_parser():
  parser = argparse.ArgumentParser(description = 'PyTorch CPN Training')
  parser.add_argument('-j', '--workers', default = 12, type = int, metavar = 'N',
                      help = 'number of data loading workers (default: 12)')
  parser.add_argument('-g', '--num_gpus', default = 1, type = int, metavar = 'N',
                      help = 'number of GPU to use (default: 1)')
  parser.add_argument('--epochs', default = 32, type = int, metavar = 'N',
                      help = 'number of total epochs to run (default: 32)')
  parser.add_argument('--start-epoch', default = 0, type = int, metavar = 'N',
                      help = 'manual epoch number (useful on restarts)')
  parser.add_argument('-c', '--checkpoint', default = 'checkpoint', type = str, metavar = 'PATH',
                      help = 'path to save checkpoint (default: checkpoint)')
  parser.add_argument('--resume', default = '', type = str, metavar = 'PATH',
                      help = 'path to latest checkpoint')
  return  parser

def add_pypath(path):
  if path not in sys.path:
    sys.path.insert(0, path)


class Config:
  cur_dir = os.path.dirname(os.path.abspath(__file__))
  this_dir_name = cur_dir.split('/')[-1]
  # root_dir = os.path.join(cur_dir, '..')
  root_dir = cur_dir
  
  model = 'CPN50'
  
  lr = 5e-4
  lr_gamma = 0.5
  lr_dec_epoch = list(range(6, 40, 6))
  
  batch_size = 1
  weight_decay = 1e-5
  
  num_class = 17
  img_path = os.path.join(root_dir, 'data', 'COCO2017', 'train2017')
  symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
  bbox_extend_factor = (0.1, 0.15)  # x, y
  
  # data augmentation setting
  scale_factor = (0.7, 1.35)
  rot_factor = 45
  
  pixel_means = np.array([122.7717, 115.9465, 102.9801])  # RGB
  data_shape = (256, 192)
  output_shape = (64, 48)
  gaussain_kernel = (7, 7)
  
  gk15 = (15, 15)
  gk11 = (11, 11)
  gk9 = (9, 9)
  gk7 = (7, 7)
  
  gt_path = os.path.join(root_dir, 'data', 'COCO2017', 'annotations', 'COCO_2017_train.json')


cfg = Config()
add_pypath(cfg.root_dir)
