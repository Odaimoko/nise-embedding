import threading
from collections import deque
import torch.multiprocessing as mp
from multiprocessing.pool import Pool
from multiprocessing import Lock
import pprint
import torch
import json
from pathlib import PurePosixPath

from nise_lib.frameitem import FrameItem
from nise_lib.nise_functions import *
from nise_lib.manager_torch import gm
from plogs.plogs import get_logger
from simple_lib.core.config import config as simple_cfg
from hr_lib.config import cfg as hr_cfg
from tron_lib.core.test_for_pt import _get_blobs
from tron_lib.core.config import cfg as tron_cfg
from tron_lib.core.config import cfg_from_file, cfg_from_list, assert_and_infer_cfg


# ─── SINGLE THREADING ───────────────────────────────────────────────────────────


def nise_pred_task_1_debug(gt_anno_dir, maskRCNN, joint_estimator, flow_model):
    # PREDICT ON TRAINING SET OF 2017
    anno_file_names = get_type_from_dir(gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    mkdir(nise_cfg.PATH.JSON_SAVE_DIR)
    mkdir(nise_cfg.PATH.DETECT_JSON_DIR)
    est_cfg = simple_cfg if nise_cfg.DEBUG.load_simple_model else hr_cfg
    
    for i, file_name in enumerate(anno_file_names):
        debug_print(i, file_name)
        if is_skip_video(nise_cfg, i, file_name):
            continue
        p = PurePosixPath(file_name)
        det_path = os.path.join(nise_cfg.PATH.DETECT_JSON_DIR, p.parts[-1])
        flow_path = os.path.join(nise_cfg.PATH.FLOW_JSON_DIR, p.stem + '.pkl')
        est_path = os.path.join(nise_cfg.PATH.DET_EST_JSON_DIR, p.stem + '.pkl')
        json_path = os.path.join(nise_cfg.PATH.JSON_SAVE_DIR, p.parts[-1])
        uni_path = os.path.join(nise_cfg.PATH.UNIFIED_JSON_DIR, p.stem + '.pkl')
        with open(file_name, 'r') as f:
            gt = json.load(f)['annolist']
        pred_frames = []
        detect_result_to_record = []
        flow_result_to_record = []
        uni_result_to_record = []
        
        # with
        if nise_cfg.DEBUG.USE_DETECTION_RESULT:
            with open(det_path, 'r')as f:
                detection_result_from_json = json.load(f)
            assert len(detection_result_from_json) == len(gt)
        else:
            detection_result_from_json = {}
        
        Q = deque(maxlen = nise_cfg.ALG._DEQUE_CAPACITY)
        for j, frame in enumerate(gt):
            # frame dict_keys(['image', 'annorect', 'imgnum', 'is_labeled', 'ignore_regions'])
            img_file_path = frame['image'][0]['name']
            img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, img_file_path)
            debug_print(j, img_file_path, indent = 1)
            annorects = frame['annorect']
            if nise_cfg.TEST.USE_GT_PEOPLE_BOX and (annorects is not None or len(annorects) != 0):
                # if only use gt bbox, then for those frames which dont have annotations, we dont estimate
                gt_joints, gt_scores = get_joints_from_annorects(annorects)
            else:
                gt_joints = torch.tensor([])
            #
            if nise_cfg.DEBUG.USE_DETECTION_RESULT:
                detect_box = detection_result_from_json[j][img_file_path]
                detect_box = torch.tensor(detect_box)
            else:
                detect_box = None
            
            fi = FrameItem(nise_cfg, est_cfg, img_file_path, 1, True, gt_joints)
            fi.detect_human(maskRCNN, detect_box)
            if nise_cfg.DEBUG.SAVE_FLOW_TENSOR:
                fi.gen_flow(flow_model, None if j == 0 else Q[-1].flow_img)
            fi.unify_bbox()
            fi.est_joints(joint_estimator)
            
            if nise_cfg.DEBUG.VISUALIZE:
                fi.assign_id_task_1_2(Q)
                threading.Thread(target = fi.visualize, args = ()).start()
            
            detect_result_to_record.append({img_file_path: fi.detect_results()})
            flow_result_to_record.append({img_file_path: fi.flow_result()})
            uni_result_to_record.append({img_file_path: fi.unfied_result()})
            
            pred_frames.append(fi.to_dict())
            Q.append(fi)
        
        with open(json_path, 'w') as f:
            json.dump({'annolist': pred_frames}, f, indent = 2)
            debug_print('pt eval json saved:', json_path)
        
        if nise_cfg.DEBUG.SAVE_DETECTION_TENSOR:
            with open(det_path, 'w') as f:
                json.dump(detect_result_to_record, f, indent = 2)
                debug_print('det results json saved:', det_path)
        if nise_cfg.DEBUG.SAVE_FLOW_TENSOR:
            torch.save(flow_result_to_record, flow_path)
            debug_print('Flow saved', flow_path)
        if nise_cfg.DEBUG.SAVE_NMS_TENSOR:
            torch.save(uni_result_to_record, uni_path)
            debug_print('NMSed boxes saved: ', uni_path)


def nise_pred_task_2_debug(gt_anno_dir, maskRCNN, joint_estimator, flow_model):
    # PREDICT ON TRAINING SET OF 2017
    anno_file_names = get_type_from_dir(gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    mkdir(nise_cfg.PATH.JSON_SAVE_DIR)
    mkdir(nise_cfg.PATH.DETECT_JSON_DIR)
    est_cfg = simple_cfg if nise_cfg.DEBUG.load_simple_model else hr_cfg
    debug_print(pprint.pformat(nise_cfg))
    for i, file_name in enumerate(anno_file_names):
        debug_print(i, file_name)
        torch.cuda.empty_cache()
        if is_skip_video(nise_cfg, i, file_name):
            debug_print('Skip', i, file_name)
            continue
        debug_print(i, file_name)
        
        p = PurePosixPath(file_name)
        json_path = os.path.join(nise_cfg.PATH.JSON_SAVE_DIR, p.parts[-1])
        with open(file_name, 'r') as f:
            gt = json.load(f)['annolist']
        pred_frames = []
        det_path = os.path.join(nise_cfg.PATH.DETECT_JSON_DIR, p.parts[-1])
        flow_path = os.path.join(nise_cfg.PATH.FLOW_JSON_DIR, p.stem + '.pkl')
        uni_path = os.path.join(nise_cfg.PATH.UNIFIED_JSON_DIR, p.stem + '.pkl')
        uni_result_to_record = []
        
        if nise_cfg.DEBUG.USE_DETECTION_RESULT:
            with open(det_path, 'r')as f:
                detection_result_from_json = json.load(f)
            assert len(detection_result_from_json) == len(gt)
        else:
            detection_result_from_json = {}
        debug_print('Precomputed detection result loaded', flow_path)
        
        if nise_cfg.DEBUG.USE_FLOW_RESULT:
            flow_result_from_json = torch.load(flow_path)
            debug_print('Precomputed flow result loaded', flow_path)
            assert len(flow_result_from_json) == len(gt)
        else:
            flow_result_from_json = {}
        
        Q = deque(maxlen = nise_cfg.ALG._DEQUE_CAPACITY)
        
        for j, frame in enumerate(gt):
            # frame dict_keys(['image', 'annorect', 'imgnum', 'is_labeled', 'ignore_regions'])
            img_file_path = frame['image'][0]['name']
            img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, img_file_path)
            debug_print(j, img_file_path, indent = 1)
            annorects = frame['annorect']
            if annorects is not None or len(annorects) != 0:
                # if only use gt bbox, then for those frames which dont have annotations, we dont estimate
                gt_joints, gt_scores = get_joints_from_annorects(annorects)
            else:
                gt_joints = torch.tensor([])
            if nise_cfg.DEBUG.USE_DETECTION_RESULT:
                detect_box = detection_result_from_json[j][img_file_path]
                detect_box = torch.tensor(detect_box)
            else:
                detect_box = None
            
            if nise_cfg.DEBUG.USE_FLOW_RESULT:
                pre_com_flow = flow_result_from_json[j][img_file_path]
            else:
                pre_com_flow = torch.tensor([])
            
            if j == 0:  # first frame doesnt have flow, joint prop
                is_first = True
            else:
                is_first = False
            fi = FrameItem(nise_cfg, est_cfg, img_file_path, is_first = is_first, gt_joints = gt_joints)
            fi.detected_boxes = detect_box
            fi.human_detected = True
            if not is_first:
                fi.flow_to_current = pre_com_flow
                fi.flow_calculated = True
                fi.joint_prop(Q)
            fi.unify_bbox()
            fi.est_joints(joint_estimator)
            pred_frames.append(fi.to_dict())
            uni_result_to_record.append({img_file_path: fi.unfied_result()})
            Q.append(fi)
        
        with open(json_path, 'w') as f:
            json.dump({'annolist': pred_frames}, f, indent = 2)
            debug_print('json saved:', json_path)
        if nise_cfg.DEBUG.SAVE_NMS_TENSOR:
            torch.save(uni_result_to_record, uni_path)
            debug_print('NMSed boxes saved: ', uni_path)


def nise_pred_task_3_debug(gt_anno_dir):
    # PREDICT ON TRAINING SET OF 2017
    anno_file_names = get_type_from_dir(gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    mkdir(nise_cfg.PATH.JSON_SAVE_DIR)
    mkdir(nise_cfg.PATH.DETECT_JSON_DIR)
    est_cfg = simple_cfg if nise_cfg.DEBUG.load_simple_model else hr_cfg
    
    for i, file_name in enumerate(anno_file_names):
        if is_skip_video(nise_cfg, i, file_name):
            debug_print('Skip', i, file_name)
            continue
        debug_print(i, file_name)
        
        p = PurePosixPath(file_name)
        json_path = os.path.join(nise_cfg.PATH.JSON_SAVE_DIR, p.parts[-1])
        with open(file_name, 'r') as f:
            gt = json.load(f)['annolist']
        pred_frames = []
        # load pre computed boxes
        uni_path = os.path.join(nise_cfg.PATH.UNI_BOX_FOR_TASK_3, p.stem + '.pkl')
        uni_boxes = torch.load(uni_path)
        
        pred_json_path = os.path.join(nise_cfg.PATH.PRED_JSON_FOR_TASK_3, p.stem + '.json')
        with open(pred_json_path, 'r') as f:
            pred = json.load(f)['annolist']
        Q = deque(maxlen = nise_cfg.ALG._DEQUE_CAPACITY)
        vis_threads = []
        for j, frame in enumerate(gt):
            img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, frame['image'][0]['name'])
            debug_print(j, img_file_path, indent = 1)
            gt_annorects = frame['annorect']
            if gt_annorects is not None and len(gt_annorects) != 0:
                # if only use gt bbox, then for those frames which dont have annotations, we dont estimate
                gt_joints, gt_scores = get_joints_from_annorects(gt_annorects)
                gt_id = torch.tensor([t['track_id'][0] for t in gt_annorects])
            else:
                gt_joints = torch.tensor([])
                gt_id = torch.tensor([])
            
            if len(Q) == 0:  # first frame doesnt have flow, joint prop
                fi = FrameItem(nise_cfg, est_cfg, img_file_path, is_first = True, task = 3, gt_joints = gt_joints,
                               gt_id = gt_id)
            else:
                fi = FrameItem(nise_cfg, est_cfg, img_file_path, task = 3, gt_joints = gt_joints, gt_id = gt_id)
            
            if nise_cfg.TEST.USE_GT_PEOPLE_BOX and gt_joints.numel() == 0:
                # we want to use gt box to debug, so ignore those which dont have annotations
                pass
            else:
                pred_annorects = pred[j]['annorect']
                if pred_annorects is not None and len(pred_annorects) != 0:
                    # if only use gt bbox, then for those frames which dont have annotations, we dont estimate
                    pred_joints, pred_joints_scores = get_joints_from_annorects(pred_annorects)
                else:
                    pred_joints = torch.tensor([])
                    pred_joints_scores = torch.tensor([])
                
                uni_box = uni_boxes[j][img_file_path]
                if nise_cfg.TEST.USE_GT_PEOPLE_BOX:
                    fi.detect_human(None, uni_box)
                else:
                    fi.detected_boxes = uni_box
                    fi.human_detected = True
                fi.joints_proped = True  # no prop here is used
                fi.unify_bbox()
                if pred_joints.numel() != 0:
                    fi.joints = pred_joints[:, :, :2]  # 3d here,
                else:
                    fi.joints = pred_joints
                fi.joints_score = pred_joints_scores
                fi.joints_detected = True
                
                if nise_cfg.TEST.ASSIGN_GT_ID:
                    fi.assign_id_using_gt_id()
                else:
                    fi.assign_id(Q)
                if nise_cfg.DEBUG.VISUALIZE:
                    t = threading.Thread(target = fi.visualize, args = ())
                    vis_threads.append(t)
                    t.start()
                Q.append(fi)
            pred_frames.append(fi.to_dict())
        
        with open(json_path, 'w') as f:
            json.dump({'annolist': pred_frames}, f, indent = 2)
            debug_print('json saved:', json_path)


def gen_fpn(gt_anno_dir, maskRCNN, joint_estimator, flow_model):
    # PREDICT ON TRAINING SET OF 2017
    anno_file_names = get_type_from_dir(gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    mkdir(nise_cfg.PATH.JSON_SAVE_DIR)
    mkdir(nise_cfg.PATH.DETECT_JSON_DIR)
    mkdir(nise_cfg.PATH.FPN_PKL_DIR)
    
    for i, file_name in enumerate(anno_file_names):
        debug_print(i, file_name)
        if is_skip_video(nise_cfg, i, file_name):
            continue
        p = PurePosixPath(file_name)
        with open(file_name, 'r') as f:
            gt = json.load(f)['annolist']
        for j, frame in enumerate(gt):
            # To save - image_scale, fmap
            if frame['is_labeled'][0]:
                fpn_path = os.path.join(nise_cfg.PATH.FPN_PKL_DIR, p.stem + '-%03d' % (j) + '.pkl')
                img_file_path = frame['image'][0]['name']
                img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, img_file_path)
                # debug_print(j, img_file_path, indent = 1)
                original_img = cv2.imread(img_file_path)  # with original size
                h, w, c = original_img.shape
                debug_print(h, w, h / w)
                break
                fmap = gen_img_fmap(tron_cfg, original_img, maskRCNN)
                torch.save(fmap, fpn_path)
                debug_print('fpn_result_to_record saved: ', fpn_path)
            
            # usage
            # maskRCNN, human_det_dataset = load_human_detect_model(human_detect_args, tron_cfg)
            #
            # t = torch.load('pre_com/fpn_pkl/000342_mpii_relpath_5sec_testsub-001.pkl')
            # fmap_boxes = get_box_fmap(maskRCNN, t, torch.tensor([
            #     [0, 4, 254, 211, 0.5],
            #     [10, 44, 554, 261, 0.5],
            #     [10, 44, 554, 211, 0.5],
            # ]))


def gen_matched_box_debug(gt_anno_dir):
    # PREDICT ON TRAINING SET OF 2017
    anno_file_names = get_type_from_dir(gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    mkdir(nise_cfg.PATH.JSON_SAVE_DIR)
    mkdir(nise_cfg.PATH.DETECT_JSON_DIR)
    debug_print(pprint.pformat(nise_cfg))
    
    est_cfg = simple_cfg if nise_cfg.DEBUG.load_simple_model else hr_cfg
    
    for i, file_name in enumerate(anno_file_names):
        if is_skip_video(nise_cfg, i, file_name):
            debug_print('Skip', i, file_name)
            continue
        debug_print(i, file_name)
        p = PurePosixPath(file_name)
        json_path = os.path.join(nise_cfg.PATH.JSON_SAVE_DIR, p.parts[-1])
        uni_path_torecord = os.path.join(nise_cfg.PATH.UNIFIED_JSON_DIR, p.stem + '.pkl')
        
        with open(file_name, 'r') as f:
            gt = json.load(f)['annolist']
        pred_frames = []
        uni_result_to_record = []
        
        # load pre computed boxes
        uni_path = os.path.join(nise_cfg.PATH.UNI_BOX_FOR_TASK_3, p.stem + '.pkl')
        uni_boxes = torch.load(uni_path)
        
        pred_json_path = os.path.join(nise_cfg.PATH.PRED_JSON_FOR_TASK_3, p.stem + '.json')
        with open(pred_json_path, 'r') as f:
            pred = json.load(f)['annolist']
        Q = deque(maxlen = nise_cfg.ALG._DEQUE_CAPACITY)
        vis_threads = []
        
        for j, frame in enumerate(gt):
            # All images
            img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, frame['image'][0]['name'])
            # debug_print(j, img_file_path, indent = 1)
            gt_annorects = frame['annorect']
            pred_annorects = pred[j]['annorect']
            gt_annorects = removeRectsWithoutPoints(gt_annorects)
            pred_annorects = removeRectsWithoutPoints(pred_annorects)
            
            prToGT, det_id = find_det_for_gt_and_assignID(gt_annorects, pred_annorects, nise_cfg)
            # if det_id.size > 0:            print(det_id)
            
            uni_box = uni_boxes[j][img_file_path]
            if pred_annorects is not None and len(pred_annorects) != 0:
                # if only use gt bbox, then for those frames which dont have annotations, we dont estimate，
                if nise_cfg.DEBUG.USE_MATCHED_JOINTS:
                    pred_joints, pred_joints_scores = get_anno_matched_joints(gt_annorects, pred_annorects, nise_cfg)
                else:
                    pred_joints, pred_joints_scores = get_joints_from_annorects(pred_annorects)
                
                if pred_joints.numel() != 0:
                    pred_joints = pred_joints[:, :, :2]  # 3d here, only position is needed
                assert (len(uni_box) == pred_joints.shape[0])
            else:
                pred_joints = torch.tensor([])
                pred_joints_scores = torch.tensor([])
            
            if prToGT.size > 0:
                # debug_print('original prtogt',prToGT)
                uni_box = uni_box[prToGT]
                pred_joints = pred_joints[prToGT]
                pred_joints_scores = pred_joints_scores[prToGT]
            
            fi = FrameItem(nise_cfg, est_cfg, img_file_path, is_first = True, task = -3)
            
            if nise_cfg.TEST.USE_GT_PEOPLE_BOX:
                fi.detect_human(None, uni_box)
            else:
                fi.detected_boxes = uni_box
                fi.human_detected = True
            fi.joints_proped = True  # no prop here is used
            fi.unify_bbox()
            fi.joints = pred_joints
            fi.joints_score = pred_joints_scores
            fi.joints_detected = True
            if nise_cfg.TEST.ASSIGN_GT_ID and det_id.size > 0:
                fi.people_ids = torch.tensor(det_id).long()
                fi.id_boxes = fi.unified_boxes
                fi.id_assigned = True
            else:
                fi.assign_id_task_1_2(Q)
            if nise_cfg.DEBUG.VISUALIZE:
                t = threading.Thread(target = fi.visualize, args = ())
                vis_threads.append(t)
                t.start()
            Q.append(fi)
            pred_frames.append(fi.to_dict())
            uni_result_to_record.append({img_file_path: fi.unfied_result()})
        
        with open(json_path, 'w') as f:
            json.dump({'annolist': pred_frames}, f, indent = 2)
            debug_print('json saved:', json_path)
        
        if nise_cfg.DEBUG.SAVE_NMS_TENSOR:
            torch.save(uni_result_to_record, uni_path_torecord)
            debug_print('NMSed boxes saved: ', uni_path_torecord)
        for t in vis_threads:
            t.join()


def gen_matched_joints(gt_anno_dir):
    # PREDICT ON TRAINING SET OF 2017
    anno_file_names = get_type_from_dir(gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    mkdir(nise_cfg.PATH.JSON_SAVE_DIR)
    mkdir(nise_cfg.PATH.DETECT_JSON_DIR)
    debug_print(pprint.pformat(nise_cfg))
    
    est_cfg = simple_cfg if nise_cfg.DEBUG.load_simple_model else hr_cfg
    
    for i, file_name in enumerate(anno_file_names):
        if is_skip_video(nise_cfg, i, file_name):
            debug_print('Skip', i, file_name)
            continue
        debug_print(i, file_name)
        p = PurePosixPath(file_name)
        json_path = os.path.join(nise_cfg.PATH.JSON_SAVE_DIR, p.parts[-1])
        uni_path_torecord = os.path.join(nise_cfg.PATH.UNIFIED_JSON_DIR, p.stem + '.pkl')
        
        with open(file_name, 'r') as f:
            gt = json.load(f)['annolist']
        pred_frames = []
        uni_result_to_record = []
        
        # load pre computed boxes
        uni_path = os.path.join(nise_cfg.PATH.UNI_BOX_FOR_TASK_3, p.stem + '.pkl')
        uni_boxes = torch.load(uni_path)
        
        pred_json_path = os.path.join(nise_cfg.PATH.PRED_JSON_FOR_TASK_3, p.stem + '.json')
        with open(pred_json_path, 'r') as f:
            pred = json.load(f)['annolist']
        Q = deque(maxlen = nise_cfg.ALG._DEQUE_CAPACITY)
        vis_threads = []
        
        for j, frame in enumerate(gt):
            # All images
            img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, frame['image'][0]['name'])
            # debug_print(j, img_file_path, indent = 1)
            gt_annorects = frame['annorect']
            pred_annorects = pred[j]['annorect']
            gt_annorects = removeRectsWithoutPoints(gt_annorects)
            pred_annorects = removeRectsWithoutPoints(pred_annorects)
            uni_box = uni_boxes[j][img_file_path]
            if pred_annorects is not None and len(pred_annorects) != 0:
                if nise_cfg.DEBUG.USE_MATCHED_JOINTS:
                    pred_joints, pred_joints_scores = get_anno_matched_joints(gt_annorects, pred_annorects, nise_cfg)
                else:
                    pred_joints, pred_joints_scores = get_joints_from_annorects(pred_annorects)
                
                if pred_joints.numel() != 0:
                    pred_joints = pred_joints[:, :, :2]  # 3d here, only position is needed
                assert (len(uni_box) == pred_joints.shape[0])
            else:
                pred_joints = torch.tensor([])
                pred_joints_scores = torch.tensor([])
            
            fi = FrameItem(nise_cfg, est_cfg, img_file_path, is_first = True, task = -3)
            fi.detected_boxes = uni_box
            fi.human_detected = True
            fi.joints_proped = True  # no prop here is used
            fi.unify_bbox()
            fi.joints = pred_joints
            fi.joints_score = pred_joints_scores
            fi.joints_detected = True
            fi.assign_id_task_1_2(Q)
            if nise_cfg.DEBUG.VISUALIZE:
                t = threading.Thread(target = fi.visualize, args = ())
                vis_threads.append(t)
                t.start()
            Q.append(fi)
            pred_frames.append(fi.to_dict())
            uni_result_to_record.append({img_file_path: fi.unfied_result()})
        
        with open(json_path, 'w') as f:
            json.dump({'annolist': pred_frames}, f, indent = 2)
            debug_print('json saved:', json_path)
        
        if nise_cfg.DEBUG.SAVE_NMS_TENSOR:
            torch.save(uni_result_to_record, uni_path_torecord)
            debug_print('NMSed boxes saved: ', uni_path_torecord)
        for t in vis_threads:
            t.join()


def task_3_with_mNet(gt_anno_dir, mNet):
    # PREDICT ON TRAINING SET OF 2017
    anno_file_names = get_type_from_dir(gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    mkdir(nise_cfg.PATH.JSON_SAVE_DIR)
    mkdir(nise_cfg.PATH.DETECT_JSON_DIR)
    est_cfg = simple_cfg if nise_cfg.DEBUG.load_simple_model else hr_cfg
    
    for i, file_name in enumerate(anno_file_names):
        if is_skip_video(nise_cfg, i, file_name):
            debug_print('Skip', i, file_name)
            continue
        debug_print(i, file_name)
        
        p = PurePosixPath(file_name)
        json_path = os.path.join(nise_cfg.PATH.JSON_SAVE_DIR, p.parts[-1])
        with open(file_name, 'r') as f:
            gt = json.load(f)['annolist']
        pred_frames = []
        # load pre computed boxes
        uni_path = os.path.join(nise_cfg.PATH.UNI_BOX_FOR_TASK_3, p.stem + '.pkl')
        uni_boxes = torch.load(uni_path)
        
        pred_json_path = os.path.join(nise_cfg.PATH.PRED_JSON_FOR_TASK_3, p.stem + '.json')
        with open(pred_json_path, 'r') as f:
            pred = json.load(f)['annolist']
        Q = deque(maxlen = nise_cfg.ALG._DEQUE_CAPACITY)
        vis_threads = []
        for j, frame in enumerate(gt):
            img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, frame['image'][0]['name'])
            debug_print(j, img_file_path, indent = 1)
            gt_annorects = frame['annorect']
            gt_joints, gt_scores = get_joints_from_annorects(gt_annorects)
            gt_id = torch.tensor(
                [t['track_id'][0] for t in gt_annorects]) if gt_annorects is not None else torch.tensor([])
            
            fi = FrameItem(nise_cfg, est_cfg, img_file_path, is_first = (len(Q) == 0),
                           task = 3, gt_joints = gt_joints, gt_id = gt_id)
            
            if nise_cfg.TEST.USE_GT_PEOPLE_BOX and gt_joints.numel() == 0:
                # we want to use gt box to debug, so ignore those which dont have annotations
                pass
            else:
                pred_annorects = pred[j]['annorect']
                pred_joints, pred_joints_scores = get_joints_from_annorects(pred_annorects)
                
                uni_box = uni_boxes[j][img_file_path]
                if nise_cfg.TEST.USE_GT_PEOPLE_BOX:
                    fi.detect_human(None, uni_box)
                else:
                    fi.detected_boxes = uni_box
                    fi.human_detected = True
                fi.joints_proped = True  # no prop here is used
                fi.unify_bbox()
                if pred_joints.numel() != 0:
                    fi.joints = pred_joints[:, :, :2]  # 3d here,
                else:
                    fi.joints = pred_joints
                fi.joints_score = pred_joints_scores
                fi.joints_detected = True
                im_p = PurePosixPath(img_file_path)
                mat_file_dir = os.path.join(nise_cfg.PATH.MNET_DIST_MAT_DIR, im_p.parts[-2])
                mat_file_path = os.path.join(mat_file_dir, im_p.stem + '.pkl')
                if os.path.exists(mat_file_path):
                    dist_mat = torch.load(mat_file_path)
                else:
                    dist_mat = None
                fi.assign_id_mNet(Q, mNet, dist_mat)
                if nise_cfg.DEBUG.VISUALIZE:
                    t = threading.Thread(target = fi.visualize, args = ())
                    vis_threads.append(t)
                    t.start()
                Q.append(fi)
            pred_frames.append(fi.to_dict())
        
        with open(json_path, 'w') as f:
            json.dump({'annolist': pred_frames}, f, indent = 2)
            debug_print('json saved:', json_path)


# ─── GENERATE TRAINING DATA FOR MATCHING NET ────────────────────────────────────

def calc_movement_videos(gt_anno_dir):
    np.set_printoptions(suppress = True)
    
    def get_joints_from_annorects(annorects):
        # Different from that in nise_functions.
        #  when a person has no joints anno, I change the operation to skip a person to init all joints to 0
        # This serves to fit the number of track_id and don't filter any joints
        all_joints = []
        for i in range(len(annorects)):
            rect = annorects[i]
            
            joints_3d = np.zeros((nise_cfg.DATA.num_joints, 3), dtype = np.float)
            points = rect['annopoints']
            # there's a person, but no annotations
            if points is None or len(points) <= 0:  # 因为有些图并没有annotation
                points = []
            else:
                points = points[0]['point']
            for pt_info in points:
                # analogous to coco.py  # matlab based on 1.
                i_pt = pt_info['id'][0]
                if 'is_visible' in pt_info.keys():  # from gt
                    joints_3d[i_pt, 0] = pt_info['x'][0] - 1 if pt_info['x'][0] > 0 else 0
                    joints_3d[i_pt, 1] = pt_info['y'][0] - 1 if pt_info['y'][0] > 0 else 0
                else:  # from pred
                    joints_3d[i_pt, 0] = pt_info['x'][0] if pt_info['x'][0] >= 0 else 0
                    joints_3d[i_pt, 1] = pt_info['y'][0] if pt_info['y'][0] >= 0 else 0
                
                joints_3d[i_pt, 2] = 1
            all_joints.append(joints_3d)
        
        joints = torch.tensor(all_joints).float()
        joints = expand_vector_to_tensor(joints)
        return joints
    
    def calc_velo(prev_joints, cur_joints):
        #  num_person x 15 x 3 (x,y,visible)
        average_velo = np.zeros([prev_joints.shape[0], cur_joints.shape[0], 3])  # joint velo and variance
        for i in range(prev_joints.shape[0]):
            for j in range(cur_joints.shape[0]):
                p = prev_joints[i, :, :2]
                c = cur_joints[j, :, :2]
                joint_vis = prev_joints[i, :, 2] * cur_joints[j, :, 2]
                
                mean_p = p[prev_joints[i, :, 2] == 1].mean(0) \
                    if len(p[prev_joints[i, :, 2] == 1]) > 0 else np.zeros(2)
                mean_c = c[cur_joints[j, :, 2] == 1].mean(0) \
                    if len(p[cur_joints[j, :, 2] == 1]) > 0 else np.zeros(2)
                average_velo[i, j, 0] = np.sqrt(((mean_p - mean_c) ** 2).sum())  # movement of center
                
                dis_to_mean_p = np.sqrt(((p - mean_p) ** 2).sum(1))
                dis_to_mean_c = np.sqrt(((c - mean_c) ** 2).sum(1))
                var_p = dis_to_mean_p[prev_joints[i, :, 2] == 1].mean(0) \
                    if len(p[prev_joints[i, :, 2] == 1]) > 0 else 0
                var_c = dis_to_mean_c[cur_joints[j, :, 2] == 1].mean(0) \
                    if len(p[cur_joints[j, :, 2] == 1]) > 0 else 0
                average_velo[i, j, 1] = var_c - var_p  # change of scale
                
                squared_diff = (p - c) * (p - c)
                dist = np.sqrt(squared_diff.sum(1))
                avg_displacement = dist.mean()
                average_velo[i, j, 2] = avg_displacement  # average movement of single joints
        
        return average_velo
    
    anno_file_names = get_type_from_dir(gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    mkdir(nise_cfg.PATH.JSON_SAVE_DIR)
    mkdir(nise_cfg.PATH.DETECT_JSON_DIR)
    mkdir(nise_cfg.PATH.FPN_PKL_DIR)
    
    mean_average_velos = []
    
    for i, file_name in enumerate(anno_file_names):
        if is_skip_video(nise_cfg, i, file_name):
            debug_print('Skip', i, file_name)
            continue
        debug_print(i, file_name)
        p = PurePosixPath(file_name)
        with open(file_name, 'r') as f:
            gt = json.load(f)['annolist']
        
        average_velos = []
        num_of_frames = 0
        prev_anno = None  # init
        # Do across frame association and calc movement
        prev_labeled = False
        for j, frame in enumerate(gt):
            if prev_labeled:
                time_elapsed = 1
            else:
                time_elapsed = 4
            prev_labeled = frame['is_labeled'][0]  # record if this is labeled
            if frame['is_labeled'][0]:
                # debug_print(j, 'labeled:\t', prev_labeled)
                anno = frame['annorect']
                num_of_frames += 1
                
                if prev_anno:
                    gt_joints = get_joints_from_annorects(anno)
                    prev_joints = get_joints_from_annorects(prev_anno)
                    
                    average_velo_mat = calc_velo(prev_joints.numpy(), gt_joints.numpy()) / time_elapsed
                    
                    gt_id = np.array([t['track_id'][0] for t in anno])
                    prev_id = np.array([t['track_id'][0] for t in prev_anno])
                    
                    dis_mat = -np.abs(
                        np.expand_dims(prev_id, 1) - np.expand_dims(gt_id, 0)
                    )
                    inds = get_matching_indices(dis_mat)
                    average_velo_vec = []
                    for i, ind in enumerate(inds):
                        (idxPr, idxGT) = ind
                        average_velo_vec.append(average_velo_mat[idxGT, idxPr])
                    if average_velo_vec:
                        average_velos.append(np.array(average_velo_vec).mean(0))
                prev_anno = anno
        
        debug_print(num_of_frames)
        average_velos = np.stack(average_velos)
        velo_vid = average_velos.mean(0)
        
        mean_average_velos.append(velo_vid)
    mean_average_velos = np.stack(mean_average_velos)
    for fn, mav in zip(anno_file_names, mean_average_velos):
        debug_print(fn, mav)


# ─── MULTI THREADING ────────────────────────────────────────────────────────────

def init_gm(_gm, n_cfg, ):
    global locks, nise_cfg
    locks = _gm
    nise_cfg = n_cfg


@log_time('validation has run')
def nise_flow_debug(gt_anno_dir, human_detector, joint_estimator, flow_model, mNet):
    # PREDICT ON POSETRACK 2017
    pp = pprint.PrettyPrinter(indent = 2)
    debug_print(pp.pformat(nise_cfg))
    
    anno_file_names = get_type_from_dir(gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    mkdir(nise_cfg.PATH.JSON_SAVE_DIR)
    # if don't pass in nise_cfg, in run_video function, nise_will be the original one
    
    est_cfg = simple_cfg if nise_cfg.DEBUG.load_simple_model else hr_cfg
    
    all_video = [(nise_cfg, est_cfg, i, file_name, human_detector, joint_estimator, flow_model, mNet) for i, file_name
                 in
                 enumerate(anno_file_names)]
    debug_print('MultiProcessing start method:', mp.get_start_method(), lvl = Levels.STATUS)
    
    # https://stackoverflow.com/questions/24941359/ctx-parameter-in-multiprocessing-queue
    # If you were to import multiprocessing.queues.Queue directly and try to instantiate it, you'll get the error you're seeing. But it should work fine if you import it from multiprocessing directly.
    
    if nise_cfg.TEST.TASK == 1:
        fun = run_one_video_task_1
    elif nise_cfg.TEST.TASK == -1:
        fun = run_one_video_flow_debug
    elif nise_cfg.TEST.TASK == -2:
        fun = run_one_video_tracking
    elif nise_cfg.TEST.TASK == -3:
        fun = oracle_id_filter
    elif nise_cfg.TEST.TASK == -4:
        fun = run_one_gen_matched_joints
    elif nise_cfg.TEST.TASK == -6:
        fun = run_one_mnet
    
    num_process = len(os.environ.get('CUDA_VISIBLE_DEVICES', default = '').split(','))
    num_process = 4
    global locks
    locks = [Lock() for _ in range(gm.gpu_num)]
    # with Pool(initializer = init_gm, initargs = (locks, nise_cfg)) as po:
    debug_print('Numprocesses', num_process)  # 这是每个 GPU 都要跑num_process 个的意思，
    # 但好像也不对。。。不是很懂这个，待看
    with Pool(processes = num_process, initializer = init_gm, initargs = (locks, nise_cfg)) as po:
        debug_print('Pool created.')
        po.starmap(fun, all_video, chunksize = 4)


@log_time('One video...')
def run_one_video_task_1(_nise_cfg, est_cfg, i: int, file_name: str, human_detector, joint_estimator, flow_model, mNet):
    '''
    Use precomputed unified boxes and joints to do matching.
    :param i:
    :param file_name:
    :param joint_estimator:
    :param flow_model:
    :return:
    '''
    torch.cuda.empty_cache()
    if is_skip_video(_nise_cfg, i, file_name):
        debug_print('Skip', i, file_name)
        return
    debug_print(i, file_name)
    
    p = PurePosixPath(file_name)
    
    det_path = os.path.join(_nise_cfg.PATH.DETECT_JSON_DIR, p.parts[-1])
    flow_path = os.path.join(_nise_cfg.PATH.FLOW_JSON_DIR, p.stem + '.pkl')
    est_path = os.path.join(_nise_cfg.PATH.DET_EST_JSON_DIR, p.stem + '.pkl')
    json_path = os.path.join(_nise_cfg.PATH.JSON_SAVE_DIR, p.parts[-1])
    uni_path = os.path.join(_nise_cfg.PATH.UNIFIED_JSON_DIR, p.stem + '.pkl')
    
    with open(file_name, 'r') as f:
        gt = json.load(f)['annolist']
    
    pred_frames = []
    detect_result_to_record = []
    flow_result_to_record = []
    est_result_to_record = []
    uni_result_to_record = []
    
    # load pre computed boxes
    
    if _nise_cfg.DEBUG.USE_DETECTION_RESULT:
        with open(det_path, 'r')as f:
            detection_result_from_json = json.load(f)
        assert len(detection_result_from_json) == len(gt)
    else:
        detection_result_from_json = {}
    
    Q = deque(maxlen = _nise_cfg.ALG._DEQUE_CAPACITY)
    vis_threads = []
    for j, frame in enumerate(gt):
        img_file_path = os.path.join(_nise_cfg.PATH.POSETRACK_ROOT, frame['image'][0]['name'])
        debug_print(j, img_file_path, indent = 1)
        gt_annorects = frame['annorect']
        if (gt_annorects is not None or len(gt_annorects) != 0):
            gt_joints, gt_scores = get_joints_from_annorects(gt_annorects)
        else:
            gt_joints = torch.tensor([])
        
        if _nise_cfg.DEBUG.USE_DETECTION_RESULT:
            detect_box = detection_result_from_json[j][img_file_path]
            detect_box = torch.tensor(detect_box)
        else:
            detect_box = None
        
        fi = FrameItem(_nise_cfg, est_cfg, img_file_path, is_first = True, task = 1, gt_joints = gt_joints)
        fi.detect_human(human_detector, detect_box)
        if _nise_cfg.DEBUG.SAVE_FLOW_TENSOR:
            fi.gen_flow(flow_model, None if j == 0 else Q[-1].flow_img)
        fi.unify_bbox()
        fi.est_joints(joint_estimator)
        
        if nise_cfg.DEBUG.VISUALIZE:
            fi.assign_id_task_1_2(Q)
            threading.Thread(target = fi.visualize, args = ()).start()
        
        detect_result_to_record.append({img_file_path: fi.detect_results()})
        flow_result_to_record.append({img_file_path: fi.flow_result()})
        est_result_to_record.append({img_file_path: fi.det_est_result()})
        uni_result_to_record.append({img_file_path: fi.unfied_result()})
        
        pred_frames.append(fi.to_dict())
    
    with open(json_path, 'w') as f:
        json.dump({'annolist': pred_frames}, f, indent = 2)
        debug_print('pt eval json saved:', json_path)
    
    if nise_cfg.DEBUG.SAVE_DETECTION_TENSOR:
        with open(det_path, 'w') as f:
            json.dump(detect_result_to_record, f, indent = 2)
            debug_print('det results json saved:', det_path)
    if nise_cfg.DEBUG.SAVE_FLOW_TENSOR:
        torch.save(flow_result_to_record, flow_path)
        debug_print('Flow saved', flow_path)
    if nise_cfg.DEBUG.SAVE_NMS_TENSOR:
        torch.save(uni_result_to_record, uni_path)
        debug_print('NMSed boxes saved: ', uni_path)
    
    for t in vis_threads:
        t.join()


@log_time('One video...')
def run_one_video_tracking(_nise_cfg, est_cfg, i: int, file_name: str, human_detector, joint_estimator,
                           flow_model, mNet):
    '''
    Use precomputed unified boxes and joints to do matching.
    :param i:
    :param file_name:
    :param joint_estimator:
    :param flow_model:
    :return:
    '''
    torch.cuda.empty_cache()
    if is_skip_video(_nise_cfg, i, file_name):
        debug_print('Skip', i, file_name)
        return
    debug_print(i, file_name)
    
    p = PurePosixPath(file_name)
    json_path = os.path.join(_nise_cfg.PATH.JSON_SAVE_DIR, p.parts[-1])
    with open(file_name, 'r') as f:
        gt = json.load(f)['annolist']
    pred_frames = []
    # load pre computed boxes
    uni_path = os.path.join(_nise_cfg.PATH.UNI_BOX_FOR_TASK_3, p.stem + '.pkl')
    uni_boxes = torch.load(uni_path)
    
    pred_json_path = os.path.join(_nise_cfg.PATH.PRED_JSON_FOR_TASK_3, p.stem + '.json')
    with open(pred_json_path, 'r') as f:
        pred = json.load(f)['annolist']
    Q = deque(maxlen = _nise_cfg.ALG._DEQUE_CAPACITY)
    vis_threads = []
    for j, frame in enumerate(gt):
        img_file_path = os.path.join(_nise_cfg.PATH.POSETRACK_ROOT, frame['image'][0]['name'])
        # debug_print(j, img_file_path, indent = 1)
        gt_annorects = frame['annorect']
        if gt_annorects is not None and len(gt_annorects) != 0:
            # if only use gt bbox, then for those frames which dont have annotations, we dont estimate
            gt_joints, gt_scores = get_joints_from_annorects(gt_annorects)
            gt_id = torch.tensor([t['track_id'][0] for t in gt_annorects])
        else:
            gt_joints = torch.tensor([])
            gt_id = torch.tensor([])
        
        if len(Q) == 0:  # first frame doesnt have flow, joint prop
            fi = FrameItem(_nise_cfg, est_cfg, img_file_path, is_first = True, task = 3, gt_joints = gt_joints,
                           gt_id = gt_id)
        else:
            fi = FrameItem(_nise_cfg, est_cfg, img_file_path, task = 3, gt_joints = gt_joints, gt_id = gt_id)
        
        if _nise_cfg.TEST.USE_GT_PEOPLE_BOX and gt_joints.numel() == 0:
            # we want to use gt box to debug, so ignore those which dont have annotations
            pass
        else:
            pred_annorects = pred[j]['annorect']
            if pred_annorects is not None and len(pred_annorects) != 0:
                # if only use gt bbox, then for those frames which dont have annotations, we dont estimate
                pred_joints, pred_joints_scores = get_joints_from_annorects(pred_annorects)
            else:
                pred_joints = torch.tensor([])
                pred_joints_scores = torch.tensor([])
            
            uni_box = uni_boxes[j][img_file_path]
            if _nise_cfg.TEST.USE_GT_PEOPLE_BOX:
                fi.detect_human(None, uni_box)
            else:
                fi.detected_boxes = uni_box
                fi.human_detected = True
            fi.joints_proped = True  # no prop here is used
            fi.unify_bbox()
            # 这里的 unify 也会排序于是也会和 joint 对不上导致 match 出错？
            
            # if _nise_cfg.TEST.USE_GT_PEOPLE_BOX:
            #     fi.joints = gt_joints
            #     fi.joints_score = gt_joints[:, :, 2]
            # else:
            if pred_joints.numel() != 0:
                fi.joints = pred_joints[:, :, :2]  # 3d here,
            else:
                fi.joints = pred_joints
            fi.joints_score = pred_joints_scores
            fi.joints_detected = True
            
            if nise_cfg.TEST.ASSIGN_GT_ID:
                fi.assign_id_using_gt_id()
            else:
                fi.assign_id(Q)
            if nise_cfg.DEBUG.VISUALIZE:
                t = threading.Thread(target = fi.visualize, args = ())
                vis_threads.append(t)
                t.start()
            Q.append(fi)
        pred_frames.append(fi.to_dict())
    
    with open(json_path, 'w') as f:
        json.dump({'annolist': pred_frames}, f, indent = 2)
        debug_print('json saved:', json_path)
    for t in vis_threads:
        t.join()

@log_time('One video...')
def run_one_mnet(_nise_cfg, est_cfg, i: int, file_name: str, human_detector, joint_estimator,
                 flow_model, mNet):
    '''
    Use precomputed unified boxes and joints to do matching.
    :param i:
    :param file_name:
    :param joint_estimator:
    :param flow_model:
    :return:
    '''
    # for rectify threads, since every thread reloads py modules and config is reset
    # FrameItem.maskRCNN = human_detector
    # human_detect_args = human_detect_parse_args()
    # cfg_from_file(human_detect_args.cfg_file)
    # tron_cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False
    
    # torch.cuda.empty_cache()
    if is_skip_video(nise_cfg, i, file_name):
        debug_print('Skip', i, file_name)
        return
    debug_print(i, file_name)
    
    p = PurePosixPath(file_name)
    json_path = os.path.join(nise_cfg.PATH.JSON_SAVE_DIR, p.parts[-1])
    with open(file_name, 'r') as f:
        gt = json.load(f)['annolist']
    pred_frames = []
    # load pre computed boxes
    uni_path = os.path.join(nise_cfg.PATH.UNI_BOX_FOR_TASK_3, p.stem + '.pkl')
    uni_boxes = torch.load(uni_path)
    
    pred_json_path = os.path.join(nise_cfg.PATH.PRED_JSON_FOR_TASK_3, p.stem + '.json')
    with open(pred_json_path, 'r') as f:
        pred = json.load(f)['annolist']
    Q = deque(maxlen = nise_cfg.ALG._DEQUE_CAPACITY)
    vis_threads = []
    for j, frame in enumerate(gt):
        img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, frame['image'][0]['name'])
        # debug_print(j, img_file_path, indent = 1)
        gt_annorects = frame['annorect']
        gt_joints, gt_scores = get_joints_from_annorects(gt_annorects)
        gt_id = torch.tensor(
            [t['track_id'][0] for t in gt_annorects]) if gt_annorects is not None else torch.tensor([])
        
        fi = FrameItem(nise_cfg, est_cfg, img_file_path, is_first = (len(Q) == 0),
                       task = 3, gt_joints = gt_joints, gt_id = gt_id)
        if nise_cfg.TEST.USE_GT_PEOPLE_BOX and gt_joints.numel() == 0:
            # we want to use gt box to debug, so ignore those which dont have annotations
            pass
        else:
            pred_annorects = pred[j]['annorect']
            pred_joints, pred_joints_scores = get_joints_from_annorects(pred_annorects)
            
            uni_box = uni_boxes[j][img_file_path]
            if nise_cfg.TEST.USE_GT_PEOPLE_BOX:
                fi.detect_human(None, uni_box)
            else:
                fi.detected_boxes = uni_box
                fi.human_detected = True
            fi.joints_proped = True  # no prop here is used
            fi.unify_bbox()
            if pred_joints.numel() != 0:
                fi.joints = pred_joints[:, :, :2]  # 3d here,
            else:
                fi.joints = pred_joints
            fi.joints_score = pred_joints_scores
            fi.joints_detected = True
            
            im_p = PurePosixPath(img_file_path)
            mat_file_dir = os.path.join(nise_cfg.PATH.MNET_DIST_MAT_DIR, im_p.parts[-2])
            mat_file_path = os.path.join(mat_file_dir, im_p.stem + '.pkl')
            if os.path.exists(mat_file_path):
                dist_mat = torch.load(mat_file_path)
            else:
                dist_mat = None
            fi.assign_id_mNet(Q, mNet, dist_mat)
            if nise_cfg.DEBUG.VISUALIZE:
                t = threading.Thread(target = fi.visualize, args = ())
                vis_threads.append(t)
                t.start()
            Q.append(fi)
        pred_frames.append(fi.to_dict())
    
    with open(json_path, 'w') as f:
        json.dump({'annolist': pred_frames}, f, indent = 2)
        debug_print('json saved:', json_path)


@log_time('One video...')
def run_one_video_flow_debug(_nise_cfg, est_cfg, i, file_name, human_detector, joint_estimator, flow_model, mNet):
    '''
    Atom function. A video be run sequentially.
    :param i:
    :param file_name:
    :param joint_estimator:
    :param flow_model:
    :return:
    '''
    torch.cuda.empty_cache()
    if is_skip_video(_nise_cfg, i, file_name):
        debug_print('Skip', i, file_name)
        return
    debug_print(i, file_name)
    
    p = PurePosixPath(file_name)
    json_path = os.path.join(_nise_cfg.PATH.JSON_SAVE_DIR, p.parts[-1])
    with open(file_name, 'r') as f:
        gt = json.load(f)['annolist']
    pred_frames = []
    det_path = os.path.join(_nise_cfg.PATH.DETECT_JSON_DIR, p.parts[-1])
    flow_path = os.path.join(_nise_cfg.PATH.FLOW_JSON_DIR, p.stem + '.pkl')
    uni_path = os.path.join(_nise_cfg.PATH.UNIFIED_JSON_DIR, p.stem + '.pkl')
    uni_result_to_record = []
    
    if _nise_cfg.DEBUG.USE_DETECTION_RESULT:
        with open(det_path, 'r')as f:
            detection_result_from_json = json.load(f)
        assert len(detection_result_from_json) == len(gt)
    else:
        detection_result_from_json = {}
    debug_print('Precomputed detection result loaded', flow_path)
    
    if _nise_cfg.DEBUG.USE_FLOW_RESULT:
        global locks
        for l in locks:
            l.acquire()
        gpu_device = gm.auto_choice()
        debug_print("Choosing GPU ", gpu_device, 'to load flow result.')
        try:
            flow_result_from_json = torch.load(flow_path, map_location = 'cuda:' + str(gpu_device))
            debug_print('Precomputed flow result loaded', flow_path)
        except Exception as e:
            print(e)
        finally:
            locks_reverse = list(locks)
            locks_reverse.reverse()
            for l in locks_reverse:
                l.release()
        
        assert len(flow_result_from_json) == len(gt)
    else:
        flow_result_from_json = {}
    
    Q = deque(maxlen = _nise_cfg.ALG._DEQUE_CAPACITY)
    
    for j, frame in enumerate(gt):
        # frame dict_keys(['image', 'annorect', 'imgnum', 'is_labeled', 'ignore_regions'])
        img_file_path = frame['image'][0]['name']
        img_file_path = os.path.join(_nise_cfg.PATH.POSETRACK_ROOT, img_file_path)
        debug_print(j, img_file_path, indent = 1)
        annorects = frame['annorect']
        if annorects is not None or len(annorects) != 0:
            # if only use gt bbox, then for those frames which dont have annotations, we dont estimate
            gt_joints, gt_scores = get_joints_from_annorects(annorects)
        else:
            gt_joints = torch.tensor([])
        if _nise_cfg.DEBUG.USE_DETECTION_RESULT:
            
            detect_box = detection_result_from_json[j][img_file_path]
            detect_box = torch.tensor(detect_box)
        else:
            detect_box = None
        
        if _nise_cfg.DEBUG.USE_FLOW_RESULT:
            pre_com_flow = flow_result_from_json[j][img_file_path]
        else:
            pre_com_flow = torch.tensor([])
        
        if j == 0 or _nise_cfg.TEST.TASK == 1:  # first frame doesnt have flow, joint prop
            fi = FrameItem(_nise_cfg, est_cfg, img_file_path, is_first = True, gt_joints = gt_joints)
            fi.detected_boxes = detect_box
            fi.human_detected = True
            fi.unify_bbox()
            fi.est_joints(joint_estimator)
        else:
            fi = FrameItem(_nise_cfg, est_cfg, img_file_path, gt_joints = gt_joints)
            fi.detected_boxes = detect_box
            fi.human_detected = True
            fi.flow_to_current = pre_com_flow
            fi.flow_calculated = True
            fi.joint_prop(Q)
            
            fi.unify_bbox()
            fi.est_joints(joint_estimator)
        pred_frames.append(fi.to_dict())
        uni_result_to_record.append({img_file_path: fi.unfied_result()})
        Q.append(fi)
    
    with open(json_path, 'w') as f:
        json.dump({'annolist': pred_frames}, f, indent = 2)
        debug_print('json saved:', json_path)
    if _nise_cfg.DEBUG.SAVE_NMS_TENSOR:
        torch.save(uni_result_to_record, uni_path)
        debug_print('NMSed boxes saved: ', uni_path)


def oracle_id_filter(_nise_cfg, est_cfg, i: int, file_name: str, human_detector, joint_estimator,
                     flow_model, mNet):
    '''
    Use precomputed unified boxes and joints to do matching.
    思路，将 gtannorect 和 det annorect 进行匹配，剩下的 det 直接输入 frameitem 作为 det 最后输出
    但是那些没有 gt 的图无法过滤掉，姑且保留所有的
    '''
    torch.cuda.empty_cache()
    if is_skip_video(_nise_cfg, i, file_name):
        debug_print('Skip', i, file_name)
        return
    debug_print(i, file_name)
    p = PurePosixPath(file_name)
    json_path = os.path.join(_nise_cfg.PATH.JSON_SAVE_DIR, p.parts[-1])
    uni_path_torecord = os.path.join(nise_cfg.PATH.UNIFIED_JSON_DIR, p.stem + '.pkl')
    
    with open(file_name, 'r') as f:
        gt = json.load(f)['annolist']
    pred_frames = []
    uni_result_to_record = []
    
    # load pre computed boxes
    uni_path = os.path.join(_nise_cfg.PATH.UNI_BOX_FOR_TASK_3, p.stem + '.pkl')
    uni_boxes = torch.load(uni_path)
    
    pred_json_path = os.path.join(_nise_cfg.PATH.PRED_JSON_FOR_TASK_3, p.stem + '.json')
    with open(pred_json_path, 'r') as f:
        pred = json.load(f)['annolist']
    Q = deque(maxlen = _nise_cfg.ALG._DEQUE_CAPACITY)
    vis_threads = []
    
    for j, frame in enumerate(gt):
        # All images
        img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, frame['image'][0]['name'])
        # debug_print(j, img_file_path, indent = 1)
        gt_annorects = frame['annorect']
        pred_annorects = pred[j]['annorect']
        gt_annorects = removeRectsWithoutPoints(gt_annorects)
        pred_annorects = removeRectsWithoutPoints(pred_annorects)
        
        prToGT, det_id = find_det_for_gt_and_assignID(gt_annorects, pred_annorects, nise_cfg)
        # if det_id.size > 0:            print(det_id)
        
        uni_box = uni_boxes[j][img_file_path]
        if pred_annorects is not None and len(pred_annorects) != 0:
            # if only use gt bbox, then for those frames which dont have annotations, we dont estimate，
            if nise_cfg.DEBUG.USE_MATCHED_JOINTS:
                pred_joints, pred_joints_scores = get_anno_matched_joints(gt_annorects, pred_annorects, nise_cfg)
            else:
                pred_joints, pred_joints_scores = get_joints_from_annorects(pred_annorects)
            
            if pred_joints.numel() != 0:
                pred_joints = pred_joints[:, :, :2]  # 3d here, only position is needed
            assert (len(uni_box) == pred_joints.shape[0])
        else:
            pred_joints = torch.tensor([])
            pred_joints_scores = torch.tensor([])
        
        if prToGT.size > 0:
            # debug_print('original prtogt',prToGT)
            uni_box = uni_box[prToGT]
            pred_joints = pred_joints[prToGT]
            pred_joints_scores = pred_joints_scores[prToGT]
        
        fi = FrameItem(nise_cfg, est_cfg, img_file_path, is_first = True, task = -3)
        
        if nise_cfg.TEST.USE_GT_PEOPLE_BOX:
            fi.detect_human(None, uni_box)
        else:
            fi.detected_boxes = uni_box
            fi.human_detected = True
        fi.joints_proped = True  # no prop here is used
        fi.unify_bbox()
        fi.joints = pred_joints
        fi.joints_score = pred_joints_scores
        fi.joints_detected = True
        if nise_cfg.TEST.ASSIGN_GT_ID and det_id.size > 0:
            fi.people_ids = torch.tensor(det_id).long()
            fi.id_boxes = fi.unified_boxes
            fi.id_assigned = True
        else:
            fi.assign_id_task_1_2(Q)
        if nise_cfg.DEBUG.VISUALIZE:
            t = threading.Thread(target = fi.visualize, args = ())
            vis_threads.append(t)
            t.start()
        Q.append(fi)
        pred_frames.append(fi.to_dict())
        uni_result_to_record.append({img_file_path: fi.unfied_result()})
    
    with open(json_path, 'w') as f:
        json.dump({'annolist': pred_frames}, f, indent = 2)
        debug_print('json saved:', json_path)
    
    if nise_cfg.DEBUG.SAVE_NMS_TENSOR:
        torch.save(uni_result_to_record, uni_path_torecord)
        debug_print('NMSed boxes saved: ', uni_path_torecord)
    for t in vis_threads:
        t.join()


def run_one_gen_matched_joints(_nise_cfg, est_cfg, i: int, file_name: str, human_detector, joint_estimator,
                               flow_model, mNet):
    '''
    Use precomputed unified boxes and joints to do matching.
    思路，将 gtannorect 和 det annorect 进行匹配，剩下的 det 直接输入 frameitem 作为 det 最后输出
    但是那些没有 gt 的图无法过滤掉，姑且保留所有的
    '''
    torch.cuda.empty_cache()
    if is_skip_video(nise_cfg, i, file_name):
        debug_print('Skip', i, file_name)
        return
    debug_print(i, file_name)
    p = PurePosixPath(file_name)
    json_path = os.path.join(nise_cfg.PATH.JSON_SAVE_DIR, p.parts[-1])
    uni_path_torecord = os.path.join(nise_cfg.PATH.UNIFIED_JSON_DIR, p.stem + '.pkl')
    
    with open(file_name, 'r') as f:
        gt = json.load(f)['annolist']
    pred_frames = []
    uni_result_to_record = []
    
    # load pre computed boxes
    uni_path = os.path.join(nise_cfg.PATH.UNI_BOX_FOR_TASK_3, p.stem + '.pkl')
    uni_boxes = torch.load(uni_path)
    
    pred_json_path = os.path.join(nise_cfg.PATH.PRED_JSON_FOR_TASK_3, p.stem + '.json')
    with open(pred_json_path, 'r') as f:
        pred = json.load(f)['annolist']
    Q = deque(maxlen = nise_cfg.ALG._DEQUE_CAPACITY)
    vis_threads = []
    
    for j, frame in enumerate(gt):
        # All images
        img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, frame['image'][0]['name'])
        # debug_print(j, img_file_path, indent = 1)
        gt_annorects = frame['annorect']
        pred_annorects = pred[j]['annorect']
        gt_annorects = removeRectsWithoutPoints(gt_annorects)
        pred_annorects = removeRectsWithoutPoints(pred_annorects)
        uni_box = uni_boxes[j][img_file_path]
        if pred_annorects is not None and len(pred_annorects) != 0:
            if nise_cfg.DEBUG.USE_MATCHED_JOINTS:
                pred_joints, pred_joints_scores = get_anno_matched_joints(gt_annorects, pred_annorects, nise_cfg)
            else:
                pred_joints, pred_joints_scores = get_joints_from_annorects(pred_annorects)
            
            if pred_joints.numel() != 0:
                pred_joints = pred_joints[:, :, :2]  # 3d here, only position is needed
            assert (len(uni_box) == pred_joints.shape[0])
        else:
            pred_joints = torch.tensor([])
            pred_joints_scores = torch.tensor([])
        
        fi = FrameItem(nise_cfg, est_cfg, img_file_path, is_first = True, task = -3)
        fi.detected_boxes = uni_box
        fi.human_detected = True
        fi.joints_proped = True  # no prop here is used
        fi.unify_bbox()
        fi.joints = pred_joints
        fi.joints_score = pred_joints_scores
        fi.joints_detected = True
        fi.assign_id_task_1_2(Q)
        if nise_cfg.DEBUG.VISUALIZE:
            t = threading.Thread(target = fi.visualize, args = ())
            vis_threads.append(t)
            t.start()
        Q.append(fi)
        pred_frames.append(fi.to_dict())
        uni_result_to_record.append({img_file_path: fi.unfied_result()})
    
    with open(json_path, 'w') as f:
        json.dump({'annolist': pred_frames}, f, indent = 2)
        debug_print('json saved:', json_path)
    
    if nise_cfg.DEBUG.SAVE_NMS_TENSOR:
        torch.save(uni_result_to_record, uni_path_torecord)
        debug_print('NMSed boxes saved: ', uni_path_torecord)
    for t in vis_threads:
        t.join()
