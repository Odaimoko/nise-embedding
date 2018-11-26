import nise_lib._init_paths
from collections import deque
import torch.backends.cudnn as cudnn

from nise_lib.nise_functions import *
from nise_lib.nise_debugging_func import *
from nise_lib.nise_config import cfg as nise_cfg
import json
from pathlib import PurePosixPath
from nise_lib.frameitem import FrameItem


def nise_pred_task_1_debug(gt_anno_dir, json_save_dir, vis_dataset, hunam_detector, joint_estimator, flow_model):
    # PREDICT ON TRAINING SET OF 2017
    anno_file_names = get_type_from_dir(
        gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    mkdir(json_save_dir)
    for i, file_name in enumerate(anno_file_names):
        print(i, file_name)
        # if not '17839_mpii' in file_name:  # the first images contains no people, cant deal with this now so ignore this.
        #     continue
        if i > 3: continue
        p = PurePosixPath(file_name)
        json_path = os.path.join(json_save_dir, p.parts[-1])
        with open(file_name, 'r') as f:
            gt = json.load(f)['annolist']
        pred_frames = []
        Q = deque(maxlen = nise_cfg.ALG._DEQUE_CAPACITY)
        for i, frame in enumerate(gt):
            # frame dict_keys(['image', 'annorect', 'imgnum', 'is_labeled', 'ignore_regions'])
            img_file_path = frame['image'][0]['name']
            img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, img_file_path)
            debug_print(img_file_path)
            if i == 0:  # first frame doesnt have flow, joint prop
                fi = FrameItem(img_file_path, 1, True)
                fi.detect_human(hunam_detector)
                fi.unify_bbox()
                fi.est_joints(joint_estimator)
                # fi.assign_id(Q)
                fi.visualize(dataset = vis_dataset)
            else:
                fi = FrameItem(img_file_path, 1)
                fi.detect_human(hunam_detector)
                # fi.gen_flow(flow_model, Q[-1].bgr_img)
                # fi.joint_prop(Q)
                fi.unify_bbox()
                fi.est_joints(joint_estimator)
                # fi.assign_id(Q, get_joints_oks_mtx)
                fi.visualize(vis_dataset)
            pred_frames.append(fi.to_dict())
            Q.append(fi)
        
        with open(json_path, 'w') as f:
            json.dump({'annolist': pred_frames}, f)


def nise_pred_task_3_debug(gt_anno_dir, json_save_dir, vis_dataset, hunam_detector, joint_estimator, flow_model):
    # PREDICT ON TRAINING SET OF 2017
    anno_file_names = get_type_from_dir(
        gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    mkdir(json_save_dir)
    for i, file_name in enumerate(anno_file_names):
        print(i, file_name)
        # if not '17839_mpii' in file_name:  # the first images contains no people, cant deal with this now so ignore this.
        #     continue
        if i>3:continue
        p = PurePosixPath(file_name)
        json_path = os.path.join(json_save_dir, p.parts[-1])
        with open(file_name, 'r') as f:
            gt = json.load(f)['annolist']
        pred_frames = []
        Q = deque(maxlen = nise_cfg.ALG._DEQUE_CAPACITY)
        for i, frame in enumerate(gt):
            # frame dict_keys(['image', 'annorect', 'imgnum', 'is_labeled', 'ignore_regions'])
            img_file_path = frame['image'][0]['name']
            img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, img_file_path)
            debug_print(img_file_path)
            if i == 0:  # first frame doesnt have flow, joint prop
                fi = FrameItem(img_file_path, True)
                fi.detect_human(hunam_detector)
                fi.unify_bbox()
                fi.est_joints(joint_estimator)
                fi.assign_id(Q)
                fi.visualize(dataset = vis_dataset)
            else:
                fi = FrameItem(img_file_path)
                fi.detect_human(hunam_detector)
                fi.gen_flow(flow_model, Q[-1].bgr_img)
                fi.joint_prop(Q)
                fi.unify_bbox()
                fi.est_joints(joint_estimator)
                fi.assign_id(Q, get_joints_oks_mtx)
                fi.visualize(vis_dataset)
            pred_frames.append(fi.to_dict())
            Q.append(fi)
        
        with open(json_path, 'w') as f:
            json.dump({'annolist': pred_frames}, f)


def nise_pred_task_3(gt_anno_dir, json_save_dir, vis_dataset, hunam_detector, joint_estimator, flow_model):
    # PREDICT ON TRAINING SET OF 2017
    anno_file_names = get_type_from_dir(
        gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    mkdir(json_save_dir)
    for file_name in anno_file_names:
        p = PurePosixPath(file_name)
        json_path = os.path.join(json_save_dir, p.parts[-1])
        with open(file_name, 'r') as f:
            gt = json.load(f)['annolist']
        pred_frames = []
        Q = deque(maxlen = nise_cfg.ALG._DEQUE_CAPACITY)
        for i, frame in enumerate(gt):
            # frame dict_keys(['image', 'annorect', 'imgnum', 'is_labeled', 'ignore_regions'])
            img_file_path = frame['image'][0]['name']
            img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, img_file_path)
            debug_print(img_file_path)
            if i == 0:  # first frame doesnt have flow, joint prop
                fi = FrameItem(img_file_path, True)
                fi.detect_human(hunam_detector)
                fi.unify_bbox()
                fi.est_joints(joint_estimator)
                fi.assign_id(Q)
                fi.visualize(dataset = vis_dataset)
            else:
                fi = FrameItem(img_file_path)
                fi.detect_human(hunam_detector)
                fi.gen_flow(flow_model, Q[-1].bgr_img)
                fi.joint_prop(Q)
                fi.unify_bbox()
                fi.est_joints(joint_estimator)
                fi.assign_id(Q, get_joints_oks_mtx)
                fi.visualize(vis_dataset)
            pred_frames.append(fi.to_dict())
            Q.append(fi)
        
        with open(json_path, 'w') as f:
            json.dump({'annolist': pred_frames}, f)

# def train_est_on_posetrack(args,config, est_model, loader, ):
#     ''''''
#     # args already reset
#     # cudnn related setting
#     cudnn.benchmark = config.CUDNN.BENCHMARK
#     cudnn.deterministic = config.CUDNN.DETERMINISTIC
#     cudnn.enabled = config.CUDNN.ENABLED
#
#     if est_model==None:
#         """train from scratch"""
