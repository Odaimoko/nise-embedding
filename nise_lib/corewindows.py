import threading
from collections import deque
import torch.multiprocessing as mp
from multiprocessing.pool import Pool
from multiprocessing import Lock
import pprint
from nise_lib.frameitem import FrameItem
from nise_lib.nise_functions import *
from nise_lib.manager_torch import gm
from plogs.plogs import get_logger


def nise_pred_task_1_debug(gt_anno_dir, vis_dataset, hunam_detector, joint_estimator, flow_model):
    # PREDICT ON TRAINING SET OF 2017
    anno_file_names = get_type_from_dir(gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    mkdir(nise_cfg.PATH.JSON_SAVE_DIR)
    mkdir(nise_cfg.PATH.DETECT_JSON_DIR)
    for i, file_name in enumerate(anno_file_names[:1]):
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
        uni_box = torch.load(
            'unifed_boxes-pre-commissioning/valid_task_-1_DETbox_allBox_propAll_propGT_tfIoU_nmsThres_0.05_0.5/' + p.stem + '.pkl')
        # uni_box = torch.load(
        #     'unifed_boxes-debug/valid_task_-1_DETbox_allBox_propAll_propGT_tfIoU_nmsThres_0.05_0.50/' + p.stem + '.pkl')
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
            #
            # if nise_cfg.DEBUG.USE_DETECTION_RESULT:
            #     detect_box = detection_result_from_json[j][img_file_path]
            #     detect_box = torch.tensor(detect_box)
            # else:
            #     detect_box = torch.tensor([])
            
            # debugging..., will delete
            detect_box = uni_box[j][img_file_path]
            
            fi = FrameItem(nise_cfg, img_file_path, 1, True, gt_joints)
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
            uni_result_to_record.append({img_file_path: fi.unfied_result()})
            
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
            torch.save(uni_result_to_record, uni_path)
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
                fi = FrameItem(nise_cfg, img_file_path, is_first = True, gt_joints = gt_joints)
                fi.detect_human(hunam_detector, detect_box)
                fi.unify_bbox()
                fi.est_joints(joint_estimator)
                fi.assign_id_task_1_2(Q)
                if nise_cfg.DEBUG.VISUALIZE:
                    fi.visualize(dataset = vis_dataset)
            else:
                fi = FrameItem(nise_cfg, img_file_path, gt_joints = gt_joints)
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


def init_gm(_gm, cfg):
    global locks, nise_cfg, nise_logger
    locks = _gm
    nise_cfg = cfg
    # nise_logger = logger


@log_time('validation 跑了')
def nise_flow_debug(gt_anno_dir, joint_estimator, flow_model):
    # PREDICT ON POSETRACK 2017
    pp = pprint.PrettyPrinter(indent = 2)
    debug_print(pp.pformat(nise_cfg))
    
    anno_file_names = get_type_from_dir(gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    mkdir(nise_cfg.PATH.JSON_SAVE_DIR)
    # if don't pass in nise_cfg, in run_video function, nise_will be the original one
    all_video = [(nise_cfg, i, file_name, joint_estimator, flow_model) for i, file_name in
                 enumerate(anno_file_names)]
    debug_print('MultiProcessing start method:', mp.get_start_method(), lvl = Levels.STATUS)
    
    # https://stackoverflow.com/questions/24941359/ctx-parameter-in-multiprocessing-queue
    # If you were to import multiprocessing.queues.Queue directly and try to instantiate it, you'll get the error you're seeing. But it should work fine if you import it from multiprocessing directly.
    
    num_process = len(os.environ.get('CUDA_VISIBLE_DEVICES', default = '').split(',')) * 3
    if num_process == 0:
        # use all devices
        num_process = 12
    global locks
    locks = [Lock() for _ in range(gm.gpu_num)]
    with Pool(1, initializer = init_gm, initargs = (locks, nise_cfg)) as po:
        debug_print('Pool created.')
        po.starmap(run_one_video_flow_debug, all_video, chunksize = 4)


@log_time('一个视频跑了')
def run_one_video_flow_debug(_nise_cfg, i, file_name, joint_estimator, flow_model):
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
    
    if _nise_cfg.DEBUG.USE_FLOW_RESULT or flow_model is None:
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
            gt_joints = get_joints_from_annorects(annorects)
        else:
            gt_joints = torch.tensor([])
        if _nise_cfg.DEBUG.USE_DETECTION_RESULT:
            
            detect_box = detection_result_from_json[j][img_file_path]
            detect_box = torch.tensor(detect_box)
        else:
            detect_box = torch.tensor([])
        
        if _nise_cfg.DEBUG.USE_FLOW_RESULT:
            pre_com_flow = flow_result_from_json[j][img_file_path]
        else:
            pre_com_flow = torch.tensor([])
        
        if j == 0:  # first frame doesnt have flow, joint prop
            fi = FrameItem(_nise_cfg, img_file_path, is_first = True, gt_joints = gt_joints)
            fi.detected_boxes = detect_box
            fi.human_detected = True
            
            fi.unify_bbox()
            fi.est_joints(joint_estimator)
        else:
            fi = FrameItem(_nise_cfg, img_file_path, gt_joints = gt_joints)
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
        json.dump({'annolist': pred_frames}, f)
        debug_print('json saved:', json_path)
    if _nise_cfg.DEBUG.SAVE_NMS_TENSOR:
        torch.save(uni_result_to_record, uni_path)
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
                fi = FrameItem(nise_cfg, img_file_path, True)
                fi.detect_human(hunam_detector)
                fi.unify_bbox()
                fi.est_joints(joint_estimator)
                fi.assign_id_task_1_2(Q)
                fi.visualize(dataset = vis_dataset)
            else:
                fi = FrameItem(nise_cfg, img_file_path)
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