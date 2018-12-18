import json
import threading
from collections import deque

from nise_lib.frameitem import FrameItem
from nise_lib.nise_functions import *


def is_skip_video(nise_cfg, i, file_name):
    s = True
    for fn in nise_cfg.TEST.ONLY_TEST:
        if fn in file_name:  # priority
            s = False
            break
    if nise_cfg.TEST.ONLY_TEST:
        return s
    if s == True:
        if i >= nise_cfg.TEST.FROM and i < nise_cfg.TEST.TO:
            s = False
    return s


def nise_pred_task_1_debug(gt_anno_dir, vis_dataset, hunam_detector, joint_estimator, flow_model):
    # PREDICT ON TRAINING SET OF 2017
    anno_file_names = get_type_from_dir(gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    mkdir(nise_cfg.PATH.JSON_SAVE_DIR)
    mkdir(nise_cfg.PATH.DETECT_JSON_DIR)
    for i, file_name in enumerate(anno_file_names):
        debug_print(i, file_name)
        if is_skip_video(nise_cfg, i, file_name):
            continue
        p = PurePosixPath(file_name)
        det_path = os.path.join(nise_cfg.PATH.DETECT_JSON_DIR, p.parts[-1])
        flow_path = os.path.join(nise_cfg.PATH.FLOW_JSON_DIR, p.stem + '.pkl')
        est_path = os.path.join(nise_cfg.PATH.DET_EST_JSON_DIR, p.stem + '.pkl')
        json_path = os.path.join(nise_cfg.PATH.JSON_SAVE_DIR, p.parts[-1])
        uni_path  = os.path.join(nise_cfg.PATH.UNIFIED_JSON_DIR, p.stem + '.pkl')
        with open(file_name, 'r') as f:
            gt = json.load(f)['annolist']
        pred_frames = []
        detect_result_to_record = []
        flow_result_to_record = []
        est_result_to_record = []
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
            if nise_cfg.TEST.USE_GT_PEOPLE_BOX and \
                    (annorects is not None or len(annorects) != 0):
                # if only use gt bbox, then for those frames which dont have annotations, we dont estimate
                gt_joints = get_joints_from_annorects(annorects)
            else:
                gt_joints = torch.tensor([])
            
            if nise_cfg.DEBUG.USE_DETECTION_RESULT:
                detect_box = detection_result_from_json[j][img_file_path]
                detect_box = torch.tensor(detect_box)
            else:
                detect_box = torch.tensor([])
            
            fi = FrameItem(img_file_path, 1, True, gt_joints)
            fi.detect_human(hunam_detector, detect_box)
            if nise_cfg.DEBUG.SAVE_FLOW_TENSOR:
                fi.gen_flow(flow_model, None if j == 0 else Q[-1].flow_img)
            fi.unify_bbox()
            fi.est_joints(joint_estimator)
            
            if nise_cfg.DEBUG.VISUALIZE:
                fi.assign_id_task_1_2(Q)
                threading.Thread(target = fi.visualize, args = [vis_dataset]).start()
                # fi.visualize(dataset = vis_dataset)
            
            detect_result_to_record.append({img_file_path: fi.detect_results()})
            flow_result_to_record.append({img_file_path: fi.flow_result()})
            est_result_to_record.append({img_file_path: fi.det_est_result()})
            uni_result_to_record.append({img_file_path:fi.unfied_result()})

            pred_frames.append(fi.to_dict())
            Q.append(fi)
        
        with open(json_path, 'w') as f:
            json.dump({'annolist': pred_frames}, f)
            debug_print('pt eval json saved:', json_path)
        
        if nise_cfg.DEBUG.SAVE_DETECTION_TENSOR:
            with open(det_path, 'w') as f:
                json.dump(detect_result_to_record, f)
                debug_print('det results json saved:', det_path)
        if nise_cfg.DEBUG.SAVE_FLOW_TENSOR:
            torch.save(flow_result_to_record, flow_path)
            debug_print('Flow saved', flow_path)
        if nise_cfg.DEBUG.SAVE_DETECTION_S_EST:
            torch.save(est_result_to_record, est_path)
            debug_print(
                'Detection\'s estimation saved', est_path
            )
        if nise_cfg.DEBUG.SAVE_NMS_TENSOR:
            torch.save(uni_result_to_record,uni_path)
            debug_print('NMSed boxes saved: ', uni_path)


def nise_pred_task_2_debug(gt_anno_dir, vis_dataset, hunam_detector, joint_estimator, flow_model):
    # PREDICT ON TRAINING SET OF 2017
    anno_file_names = get_type_from_dir(
        gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    mkdir(nise_cfg.PATH.JSON_SAVE_DIR)
    for i, file_name in enumerate(anno_file_names):
        debug_print(i, file_name)
        if is_skip_video(nise_cfg, i, file_name): continue
        
        p = PurePosixPath(file_name)
        json_path = os.path.join(nise_cfg.PATH.JSON_SAVE_DIR, p.parts[-1])
        with open(file_name, 'r') as f:
            gt = json.load(f)['annolist']
        pred_frames = []
        det_path = os.path.join(nise_cfg.PATH.DETECT_JSON_DIR, p.parts[-1])
        flow_path = os.path.join(nise_cfg.PATH.FLOW_JSON_DIR, p.stem + '.pkl')
        est_path = os.path.join(nise_cfg.PATH.DET_EST_JSON_DIR, p.stem + '.pkl')
        
        if nise_cfg.DEBUG.USE_DETECTION_RESULT:
            with open(det_path, 'r')as f:
                detection_result_from_json = json.load(f)
            assert len(detection_result_from_json) == len(gt)
        else:
            detection_result_from_json = {}
        
        if nise_cfg.DEBUG.USE_FLOW_RESULT or flow_model is None:
            flow_result_from_json = torch.load(flow_path)
            assert len(flow_result_from_json) == len(gt)
        else:
            flow_result_from_json = {}
        
        if nise_cfg.DEBUG.USE_DET_EST_RESULT or joint_estimator is None:
            det_est_result_from_json = torch.load(est_path)
            assert len(det_est_result_from_json) == len(gt)
        else:
            det_est_result_from_json = {}
        
        Q = deque(maxlen = nise_cfg.ALG._DEQUE_CAPACITY)
        for j, frame in enumerate(gt):
            # frame dict_keys(['image', 'annorect', 'imgnum', 'is_labeled', 'ignore_regions'])
            img_file_path = frame['image'][0]['name']
            img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, img_file_path)
            debug_print(j, img_file_path, indent = 1)
            annorects = frame['annorect']
            if annorects is not None or len(annorects) != 0:
                # if only use gt bbox, then for those frames which dont have annotations, we dont estimate
                gt_joints = get_joints_from_annorects(annorects)
            else:
                gt_joints = torch.tensor([])
            if nise_cfg.DEBUG.USE_DETECTION_RESULT:
                
                detect_box = detection_result_from_json[j][img_file_path]
                detect_box = torch.tensor(detect_box)
            else:
                detect_box = torch.tensor([])
            
            if nise_cfg.DEBUG.USE_FLOW_RESULT:
                pre_com_flow = flow_result_from_json[j][img_file_path]
            else:
                pre_com_flow = torch.tensor([])
            
            if nise_cfg.DEBUG.USE_DET_EST_RESULT:
                pre_com_det_est = det_est_result_from_json[j][img_file_path]
            else:
                pre_com_det_est = torch.tensor([])
            
            if j == 0:  # first frame doesnt have flow, joint prop
                fi = FrameItem(img_file_path, is_first = True, gt_joints = gt_joints)
                fi.detect_human(hunam_detector, detect_box)
                fi.unify_bbox()
                fi.est_joints(joint_estimator)
                fi.assign_id_task_1_2(Q)
                if nise_cfg.DEBUG.VISUALIZE:
                    fi.visualize(dataset = vis_dataset)
            else:
                fi = FrameItem(img_file_path, gt_joints = gt_joints)
                fi.detect_human(hunam_detector, detect_box)
                if nise_cfg.DEBUG.USE_FLOW_RESULT:
                    fi.flow_to_current = pre_com_flow
                else:
                    fi.gen_flow(flow_model, Q[-1].flow_img)
                if nise_cfg.DEBUG.USE_DET_EST_RESULT:
                    fi.joint_prop(Q)
                else:
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


def nise_flow_debug(gt_anno_dir, vis_dataset, hunam_detector, joint_estimator, flow_model):
    # PREDICT ON TRAINING SET OF 2017
    anno_file_names = get_type_from_dir(
        gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    mkdir(nise_cfg.PATH.JSON_SAVE_DIR)
    for i, file_name in enumerate(anno_file_names):
        debug_print(i, file_name)
        if is_skip_video(nise_cfg, i, file_name): continue
        
        p = PurePosixPath(file_name)
        json_path = os.path.join(nise_cfg.PATH.JSON_SAVE_DIR, p.parts[-1])
        with open(file_name, 'r') as f:
            gt = json.load(f)['annolist']
        pred_frames = []
        det_path = os.path.join(nise_cfg.PATH.DETECT_JSON_DIR, p.parts[-1])
        flow_path = os.path.join(nise_cfg.PATH.FLOW_JSON_DIR, p.stem + '.pkl')
        est_path = os.path.join(nise_cfg.PATH.DET_EST_JSON_DIR, p.stem + '.pkl')
        uni_path  = os.path.join(nise_cfg.PATH.UNIFIED_JSON_DIR, p.stem + '.pkl')
        
        uni_result_to_record = []

        
        if nise_cfg.DEBUG.USE_DETECTION_RESULT:
            with open(det_path, 'r')as f:
                detection_result_from_json = json.load(f)
            assert len(detection_result_from_json) == len(gt)
        else:
            detection_result_from_json = {}
        
        if nise_cfg.DEBUG.USE_FLOW_RESULT or flow_model is None:
            flow_result_from_json = torch.load(flow_path)
            debug_print('Precomputed flow result loaded', flow_path)
            assert len(flow_result_from_json) == len(gt)
        else:
            flow_result_from_json = {}
        
        if nise_cfg.DEBUG.USE_DET_EST_RESULT or joint_estimator is None:
            det_est_result_from_json = torch.load(est_path)
            assert len(det_est_result_from_json) == len(gt)
        else:
            det_est_result_from_json = {}
        
        Q = deque(maxlen = nise_cfg.ALG._DEQUE_CAPACITY)
        for j, frame in enumerate(gt):
            # frame dict_keys(['image', 'annorect', 'imgnum', 'is_labeled', 'ignore_regions'])
            img_file_path = frame['image'][0]['name']
            img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, img_file_path)
            debug_print(j, img_file_path, indent = 1)
            annorects = frame['annorect']
            if annorects is not None or len(annorects) != 0:
                # if only use gt bbox, then for those frames which dont have annotations, we dont estimate
                gt_joints = get_joints_from_annorects(annorects)
            else:
                gt_joints = torch.tensor([])
            if nise_cfg.DEBUG.USE_DETECTION_RESULT:
                
                detect_box = detection_result_from_json[j][img_file_path]
                detect_box = torch.tensor(detect_box)
            else:
                detect_box = torch.tensor([])
            
            if nise_cfg.DEBUG.USE_FLOW_RESULT:
                pre_com_flow = flow_result_from_json[j][img_file_path]
            else:
                pre_com_flow = torch.tensor([])
            
            if nise_cfg.DEBUG.USE_DET_EST_RESULT:
                pre_com_det_est = det_est_result_from_json[j][img_file_path]
            else:
                pre_com_det_est = torch.tensor([])
            
            if j == 0:  # first frame doesnt have flow, joint prop
                fi = FrameItem(img_file_path, is_first = True, gt_joints = gt_joints)
                fi.detected_boxes = detect_box
                fi.human_detected = True
                
                fi.unify_bbox()
                fi.est_joints(joint_estimator)
            else:
                fi = FrameItem(img_file_path, gt_joints = gt_joints)
                fi.detected_boxes = detect_box
                fi.human_detected = True
                fi.flow_to_current = pre_com_flow
                fi.flow_calculated = True
                fi.joint_prop(Q)
                
                fi.unify_bbox()
                fi.est_joints(joint_estimator)
            pred_frames.append(fi.to_dict())
            uni_result_to_record.append({img_file_path:fi.unfied_result()})
            Q.append(fi)
        
        with open(json_path, 'w') as f:
            json.dump({'annolist': pred_frames}, f)
            debug_print('json saved:', json_path)
        if nise_cfg.DEBUG.SAVE_NMS_TENSOR:
            torch.save(uni_result_to_record,uni_path)
            debug_print('NMSed boxes saved: ', uni_path)

def nise_pred_task_3(gt_anno_dir, vis_dataset, hunam_detector, joint_estimator, flow_model):
    # PREDICT ON TRAINING SET OF 2017
    anno_file_names = get_type_from_dir(
        gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    mkdir(nise_cfg.PATH.JSON_SAVE_DIR)
    for file_name in anno_file_names:
        p = PurePosixPath(file_name)
        json_path = os.path.join(nise_cfg.PATH.JSON_SAVE_DIR, p.parts[-1])
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
