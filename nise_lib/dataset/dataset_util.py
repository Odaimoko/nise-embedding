import threading
from collections import deque
import torch.multiprocessing as mp
from multiprocessing.pool import Pool
from multiprocessing import Lock
import pprint
import torch
import json
from pathlib import PurePosixPath

from nise_lib.nise_functions import *


# @log_time("Getting joints heatmap...")
def gen_joints_hmap(union_box_size, hmap_size, joints, joint_scores):
    '''
    All op is writen in numpy, and converted to tensor at the end
        # copy from simple-baseline

    :param union_box_size: W,H
    :param hmap_size: (h,w)
    :param joints: 15x2
    :param joint_scores: (15,)
    :return:
    '''
    num_joints = joints.shape[0]
    
    target = np.zeros((num_joints,
                       hmap_size[1],
                       hmap_size[0]),
                      dtype = np.float32)  # nj x w x h ???
    target_weight = joint_scores  # not visibility since all is visible
    
    feat_stride = union_box_size / hmap_size
    radius = nise_cfg.TRAIN.JOINT_MAP_SIGMA * 3
    for joint_id in range(num_joints):
        # joint coord is [x,y]
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        ul = [int(mu_x - radius), int(mu_y - radius)]  # upper left
        br = [int(mu_x + radius + 1), int(mu_y + radius + 1)]  # bottom right
        
        size = 2 * radius + 1  # 13
        x = np.arange(0, size, 1, np.float32)  # (13,)
        y = x[:, np.newaxis]  # (1,13)
        x0 = y0 = size // 2  # 6
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * radius ** 2))
        
        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], hmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], hmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], hmap_size[0])
        img_y = max(0, ul[1]), min(br[1], hmap_size[1])
        
        v = target_weight[joint_id]
        target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]] * nise_cfg.TRAIN.JOINT_MAP_SCALE \
            * v
        # cv2.imwrite('joint_%2d.jpg' % joint_id, target[joint_id])
    return to_torch(target)


def get_one_sample(p_fmap_pkl, c_fmap_pkl, u, p_box_idx, c_box_idx,
                   p_pred_joints, p_pred_joints_scores, c_pred_joints, c_pred_joints_scores):
    start = time.time()
    p_fmap = get_box_fmap(p_fmap_pkl, u)
    c_fmap = get_box_fmap(c_fmap_pkl, u)
    # debug_print('get fmap', time.time() - start)
    # load joints
    p_joints, p_joints_scores = p_pred_joints[p_box_idx, :, :2], p_pred_joints_scores[p_box_idx, :]
    c_joints, c_joints_scores = c_pred_joints[c_box_idx, :, :2], c_pred_joints_scores[c_box_idx, :]
    
    start = time.time()
    # gen joints hmap
    bs, C, mH, mW = p_fmap.shape
    u_box_size = torch.tensor([
        u[0, 2] - u[0, 0],  # W
        u[0, 3] - u[0, 1],  # H
    ]).numpy()
    
    # convert joint pos to be relative to union box, some may be less than 0
    def get_union_box_based_joints(u, joints):
        box_size = u_box_size.tolist()
        joints = joints.clone()
        joints[:, 0] -= u[0, 0]
        joints[:, 1] -= u[0, 1]
        joints[:, 0].clamp_(0, box_size[0])
        joints[:, 1].clamp_(0, box_size[1])
        return joints
    
    p_joints = get_union_box_based_joints(u, p_joints)
    c_joints = get_union_box_based_joints(u, c_joints)
    
    p_joints_hmap = gen_joints_hmap(u_box_size,
                                    (mH, mW),
                                    p_joints, p_joints_scores)
    c_joints_hmap = gen_joints_hmap(u_box_size,
                                    (mH, mW),
                                    c_joints, c_joints_scores)
    # debug_print('get joints hemap ', time.time() - start)
    
    inputs = torch.cat([p_fmap.squeeze(), c_fmap.squeeze(), p_joints_hmap, c_joints_hmap])
    
    return inputs


# @log_time('Generating all inputs...')
def gen_all_inputs(all_inputs, all_samples, p_fmap_pkl, c_fmap_pkl,
                   p_pred_joints, p_pred_joints_scores, c_pred_joints, c_pred_joints_scores):
    # since when generating all training samples, we'll fill in all zeros if no enough samples
    assert len(all_inputs) >= len(all_samples)
    for i, s in enumerate(all_samples):
        p_box_idx, c_box_idx, union_box = s
        all_inputs[i] = get_one_sample(p_fmap_pkl, c_fmap_pkl, union_box,
                                       p_box_idx, c_box_idx,
                                       p_pred_joints, p_pred_joints_scores,
                                       c_pred_joints, c_pred_joints_scores)
    return all_inputs
