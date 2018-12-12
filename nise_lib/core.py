import nise_lib._init_paths
from collections import deque
import torch.backends.cudnn as cudnn

from nise_lib.nise_functions import *
from nise_lib.nise_debugging_func import *
import json
from pathlib import PurePosixPath
from nise_lib.frameitem import FrameItem

from nise_lib.nise_config import nise_cfg


def is_skip_video(nise_cfg, i, file_name):
    s = True
    for fn in nise_cfg.TEST.ONLY_TEST:
        if fn in file_name:
            s = False
            break
    if s == True:
        if i >= nise_cfg.TEST.FROM and i < nise_cfg.TEST.TO:
            s = False
    return s


def nise_pred_task_1_debug(gt_anno_dir, json_save_dir, vis_dataset, hunam_detector, joint_estimator, flow_model):
    # PREDICT ON TRAINING SET OF 2017
    anno_file_names = get_type_from_dir(gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    mkdir(json_save_dir)
    for i, file_name in enumerate(anno_file_names):
        
        debug_print(i, file_name)
        if is_skip_video(nise_cfg, i, file_name): continue
        p = PurePosixPath(file_name)
        json_path = os.path.join(json_save_dir, p.parts[-1])
        with open(file_name, 'r') as f:
            gt = json.load(f)['annolist']
        pred_frames = []
        Q = deque(maxlen = nise_cfg.ALG._DEQUE_CAPACITY)
        for j, frame in enumerate(gt):
            # frame dict_keys(['image', 'annorect', 'imgnum', 'is_labeled', 'ignore_regions'])
            img_file_path = frame['image'][0]['name']
            img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, img_file_path)
            debug_print(j, img_file_path, indent = 1)
            annorects = frame['annorect']
            if nise_cfg.TEST.USE_GT_VALID_BOX and \
                    (annorects is not None or len(annorects) != 0):
                # if only use gt bbox, then for those frames which dont have annotations, we dont estimate
                gt_joints = get_joints_from_annorects(annorects)
            else:
                gt_joints = torch.tensor([])
            
            fi = FrameItem(img_file_path, 1, True)
            fi.detect_human(hunam_detector, gt_joints)
            fi.unify_bbox()
            fi.est_joints(joint_estimator)
            
            if nise_cfg.DEBUG.VISUALIZE:
                fi.assign_id_task_1_2(Q)
                fi.visualize(dataset = vis_dataset)
            pred_frames.append(fi.to_dict())
            Q.append(fi)
        
        with open(json_path, 'w') as f:
            json.dump({'annolist': pred_frames}, f)
            debug_print('json saved:', json_path)


def nise_pred_task_3_debug(gt_anno_dir, json_save_dir, vis_dataset, hunam_detector, joint_estimator, flow_model):
    # PREDICT ON TRAINING SET OF 2017
    anno_file_names = get_type_from_dir(
        gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    mkdir(json_save_dir)
    for i, file_name in enumerate(anno_file_names):
        debug_print(i, file_name)
        if is_skip_video(nise_cfg, i, file_name): continue

        p = PurePosixPath(file_name)
        json_path = os.path.join(json_save_dir, p.parts[-1])
        with open(file_name, 'r') as f:
            gt = json.load(f)['annolist']
        pred_frames = []
        Q = deque(maxlen = nise_cfg.ALG._DEQUE_CAPACITY)
        for j, frame in enumerate(gt):
            # frame dict_keys(['image', 'annorect', 'imgnum', 'is_labeled', 'ignore_regions'])
            img_file_path = frame['image'][0]['name']
            img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, img_file_path)
            debug_print(j, img_file_path, indent = 1)
            if j == 0:  # first frame doesnt have flow, joint prop
                fi = FrameItem(img_file_path, is_first = True)
                fi.detect_human(hunam_detector)
                fi.unify_bbox()
                fi.est_joints(joint_estimator)
                fi.assign_id_task_1_2(Q)
                if nise_cfg.DEBUG.VISUALIZE:
                    fi.visualize(dataset = vis_dataset)
            else:
                fi = FrameItem(img_file_path)
                fi.detect_human(hunam_detector)
                fi.gen_flow(flow_model, Q[-1].flow_img)
                fi.joint_prop(Q)
                fi.unify_bbox()
                fi.est_joints(joint_estimator)
                fi.assign_id_task_1_2(Q, get_joints_oks_mtx)
                if nise_cfg.DEBUG.VISUALIZE:
                    fi.visualize(vis_dataset)
            pred_frames.append(fi.to_dict())
            Q.append(fi)
        
        with open(json_path, 'w') as f:
            json.dump({'annolist': pred_frames}, f)
            debug_print('json saved:', json_path)


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
                fi.assign_id_task_1_2(Q)
                fi.visualize(dataset = vis_dataset)
            else:
                fi = FrameItem(img_file_path)
                fi.detect_human(hunam_detector)
                fi.gen_flow(flow_model, Q[-1].bgr_img)
                fi.joint_prop(Q)
                fi.unify_bbox()
                fi.est_joints(joint_estimator)
                fi.assign_id_task_1_2(Q, get_joints_oks_mtx)
                fi.visualize(vis_dataset)
            pred_frames.append(fi.to_dict())
            Q.append(fi)
        
        with open(json_path, 'w') as f:
            json.dump({'annolist': pred_frames}, f)

