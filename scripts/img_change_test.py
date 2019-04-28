#
# ──────────────────────────────────────────────────────────────────────────────────── I ──────────
#   :::::: U S E   P R E T R A I N E D   M O D E L S   T O   T E S T
#       R E - I M P L E M E N T E D  S I M P L E   B A S E L I N E :
# ────────────────────────────────────────────────────────────────────────────

from pathlib import PurePosixPath
import argparse
import pprint
import torch.multiprocessing as mp
import warnings
from torch import optim
from collections import OrderedDict
# local packages
import init_paths
from nise_lib.nise_config import nise_cfg, nise_logger
from nise_lib.nise_functions import *
import imgaug as ia
from nise_lib.core import *

image = cv2.imread('data/pt17/images/bonn_mpii_train_5sec/00231_mpii/00000061.jpg')

cv2.imwrite('im_contrast.jpg', random_contrast(image, 10))

mb = ia.augmenters.blur.MotionBlur(40, 75, .8, 0)
img = mb.augment_image(image)
cv2.imwrite('im_blur.jpg', img)
