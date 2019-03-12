import sys

sys.path.append('..')
import scipy
from nise_lib.nise_functions import *
from nise_lib.nise_debugging_func import *

from nise_utils.simple_vis import get_batch_image_with_joints, \
    save_single_whole_image_with_joints
import tron_lib.utils.vis as vis_utils
from tron_lib.core.test_for_pt import im_detect_all
from simple_lib.core.inference import get_final_preds
from simple_lib.core.config import config as simple_cfg
# from nise_utils.cpn_transforms import get_affine_transform
from nise_utils.simple_transforms import flip_back, get_affine_transform


class FrameItem:
    max_id = 0
    '''
        Item in Deque, storing image (for flow generation), joints(for propagation),
         trackids.
         One image or a batch of images?
    '''
    
    # @log_time('\tInit FI……')
    def __init__(self, nise_cfg, simple_cfg, img_path, task = 2, is_first = False, gt_joints = None, gt_id = None):
        '''

        :param is_first: is this frame the first of the sequence
        '''
        self.cfg = nise_cfg
        self.task = task  # 1 for single-frame, 2 for multi-frame and tracking
        self.is_first = is_first
        self.img_path = img_path
        self.img_name = PurePosixPath(img_path).stem
        # tensor with size C x H x W,  detector needs BGR so here use bgr.
        
        self.original_img = cv2.imread(self.img_path)  # with original size
        self.ori_img_h, self.ori_img_w, _ = self.original_img.shape
        # 依然是W/H， 回忆model第一个192是宽。
        self.joint_est_mode_size = simple_cfg.MODEL.IMAGE_SIZE
        self.img_ratio = self.joint_est_mode_size[0] / self.joint_est_mode_size[1]
        
        if self.cfg.TEST.USE_GT_PEOPLE_BOX and self.task == 1:
            # no use, just rectify
            self.flow_img = self.original_img
        else:
            if self.cfg.ALG.FLOW_MODE == self.cfg.ALG.FLOW_PADDING_END:
                multiple = self.cfg.ALG.FLOW_MULTIPLE
                h, w, _ = self.original_img.shape
                extra_right = (multiple - (w % multiple)) % multiple
                extra_bottom = (multiple - (h % multiple)) % multiple
                
                self.flow_img = np.pad(self.original_img, ((0, extra_bottom), (0, extra_right), (0, 0)),
                                       mode = 'constant', constant_values = 0)
            else:
                self.flow_img = scipy.misc.imresize(  # h and w is reversed
                    self.original_img,
                    (self.cfg.DATA.flow_input_size[1], self.cfg.DATA.flow_input_size[0])
                )
        
        if self.cfg.ALG.FLOW_MODE == self.cfg.ALG.FLOW_PADDING_END:
            self.img_outside_flow = self.original_img
        else:
            self.img_outside_flow = self.flow_img
        
        # only used in computation of center and scale
        self.img_h, self.img_w, _ = self.img_outside_flow.shape
        
        if gt_joints is not None and gt_joints.numel() != 0:
            
            gt_box = joints_to_boxes(gt_joints[:, :, :2], gt_joints[:, :, 2], (self.img_w, self.img_h))
            gt_box = expand_vector_to_tensor(gt_box)
            gt_scores = torch.ones([gt_box.shape[0]])
            gt_scores.unsqueeze_(1)
            self.gt_boxes, idx = filter_bbox_with_area(torch.cat([gt_box, gt_scores], -1))
            self.gt_joints = expand_vector_to_tensor(gt_joints[idx, :], 3)
        else:
            self.gt_boxes = torch.tensor([])
            self.gt_joints = torch.tensor([])
        if gt_id is not None:  # only used in assign_id_using_gt_id
            self.gt_id = gt_id
        else:
            self.gt_id = torch.tensor([])
        
        # tensor num_person x num_joints x 2
        # should be LongTensor since we are going to use them as index. same length with unified_bbox
        self.joints = torch.tensor([])
        self.joints_score = torch.tensor([])
        self.new_joints = torch.tensor([])  # for similarity
        self.new_joints_vis = torch.tensor([])  # only used using gt box
        self.flow_to_current = None  # 2, h, w
        
        # coord in the original image
        self.detected_boxes = torch.tensor([])  # num_people x 5, with score
        self.prop_boxes = torch.tensor([])
        self.unified_boxes = torch.tensor([])
        self.id_boxes = torch.tensor([])  # those have ids. should have the same length with human_ids
        self.id_idx_in_unified = torch.tensor([])
        
        self.people_ids = torch.tensor([])
        self.pt_eval_dict = {}
        
        # ─── FLAGS FOR CORRENT ORDER ─────────────────────────────────────
        self.human_detected = False
        self.flow_calculated = False
        self.joints_proped = False
        self.boxes_unified = False
        self.joints_detected = False
        self.id_assigned = False
        
        self.NO_BOXES = False
    
    # def get_box_from_joints
    
    # @log_time('\t人检测……')
    def detect_human(self, detector, prepared_detection = None):
        '''
        :param detector:
        :return: people is represented as tensor of size num_people x 4. The result is NMSed.
        '''
        
        if self.cfg.TEST.USE_GT_PEOPLE_BOX:
            self.detected_boxes = self.gt_boxes
        elif prepared_detection is not None:  # 0 is 0 and prepared_detection.numel() >= 5:
            self.detected_boxes = prepared_detection
        else:
            cls_boxes = im_detect_all(detector, self.original_img)  # the example from detectron use this in this way
            human_bboxes = torch.from_numpy(cls_boxes[1])  # people is the first class of coco， 0 for background
            self.detected_boxes = human_bboxes
        
        if self.cfg.ALG.FILTER_HUMAN_WHEN_DETECT:
            # Dont use all, only high confidence
            self.detected_boxes, _ = filter_bbox_with_scores(self.detected_boxes)
            # 有可能出现没检测到的情况，大部分不可能，但是工程起见。
            # 2TODO 比如 bonn_mpii_train_5sec\00098_mpii 最前面几个是全黑所以检测不到……emmmm怎么办呢
            self.detected_boxes = expand_vector_to_tensor(self.detected_boxes)
        
        # debug_print('Detected', self.detected_boxes.shape[0], 'boxes', indent = 1)
        self.human_detected = True
    
    # @log_time('\t生成flow……')
    def gen_flow(self, flow_model, prev_frame_img = None):
        """ - Normalize people crop to some size to fit flow input
        
        # data[0] torch.Size([8, 3, 2, 384, 1024]) ，bs x channels x num_images, h, w
        # target torch.Size([8, 2, 384, 1024]) maybe offsets
        
        """
        # No precedent functions, root
        if prev_frame_img is None:  # if this is first frame
            return
        """ - Now we have some images Q and current img, calculate FLow using previous and current one - """
        resized_img = self.flow_img[:, :, ::-1]  # to rgb# flow uses rgb
        resized_prev = prev_frame_img[:, :, ::-1]
        resized_img = im_to_torch(resized_img)
        resized_prev = im_to_torch(resized_prev)
        ti = torch.stack([resized_prev, resized_img], 1)
        self.flow_to_current = pred_flow(ti, flow_model)
        self.flow_to_current = self.flow_to_current[:, :self.ori_img_h, :self.ori_img_w]
        self.flow_calculated = True
    
    # @log_time('\tJoint prop……')
    def joint_prop(self, Q, use_pre_com_det_est = False, pre_com_det_est = None):
        '''
         WHAT IS PROP BOX SCORE???
        :param prev_frame_joints: 2 x num_people x num_joints
        :return:
        '''
        if self.is_first: return
        if not self.flow_calculated:
            raise ValueError('Flow not calculated yet.')
        """ - choose prev joints pos, add flow to determine pos in the next frame; generate new bbox - """
        
        def set_empty_joint():
            self.joints_proped = True
        
        def check_joints_format(prev_frame_joints):
            prev_frame_joints = expand_vector_to_tensor(prev_frame_joints, 3)
            if not ((prev_frame_joints.shape[2] == self.cfg.DATA.num_joints and prev_frame_joints.shape[0] == 2) or (
                    prev_frame_joints.shape[2] == 2 and prev_frame_joints.shape[1] == self.cfg.DATA.num_joints)):
                raise ValueError(
                    'Size not matched, current size ' + str(prev_frame_joints.shape))
            if prev_frame_joints.shape[2] == self.cfg.DATA.num_joints:
                # :param prev_frame_joints: 2x num_people x num_joints. to num_people x num_joints x 2
                prev_frame_joints = torch.transpose(prev_frame_joints, 0, 1)
                prev_frame_joints = torch.transpose(prev_frame_joints, 1, 2)
            return prev_frame_joints
        
        prev_frame = Q[-1]
        prev_frame_joints = prev_frame.joints
        prev_frame_joints_scores = prev_frame.joints_score
        # preprocess
        if self.cfg.TEST.USE_ALL_GT_JOINTS_TO_PROP and prev_frame.gt_boxes.numel() != 0:
            # dont care about prev_joints
            prev_frame_joints = prev_frame.gt_joints[:, :, :2]
            prev_frame_joints_vis = prev_frame.gt_joints[:, :, 2]
            prev_box_scores = torch.ones([prev_frame_joints.shape[0]]) * self.cfg.TEST.GT_BOX_SCORE
            prev_frame_joints_scores = prev_frame_joints_vis  # visible, score=1; not visible, score=0
        elif prev_frame_joints.numel() != 0:
            # if there are joints to propagate, use filtered ones
            if self.cfg.ALG.JOINT_PROP_WITH_FILTERED_HUMAN:
                prev_filtered_box, filtered_idx = prev_frame.get_filtered_bboxes_with_thres(
                    thres = self.cfg.ALG.PROP_HUMAN_THRES)
                if prev_filtered_box.numel() == 0:
                    set_empty_joint()
                    debug_print('Proped', 0, 'boxes', indent = 1)
                    return
                prev_frame_joints = prev_frame_joints[filtered_idx]
                # 2TODO: 如果 score 是一个标量，在242行就不能在 dim=1的时候 unsqueeze，只有 score 是一个向量才行
                prev_box_scores = prev_frame.unified_boxes[filtered_idx, -1]
                prev_frame_joints_scores = prev_frame_joints_scores[filtered_idx]
            else:  # use all prev joints rather than filtered, or should we?
                prev_filtered_box = prev_frame.unified_boxes
                prev_box_scores = prev_frame.unified_boxes[:, -1]
            prev_filtered_box = expand_vector_to_tensor(prev_filtered_box)
            prev_frame_joints = expand_vector_to_tensor(prev_frame_joints, 3)
            # Todo is visibility right?
            prev_frame_joints_vis = torch.ones(prev_frame_joints.shape)[:, :, 0]  # numpeople x numjoints,
            prev_frame_joints_scores = expand_vector_to_tensor(prev_frame_joints_scores)
            
            if self.cfg.TEST.USE_GT_JOINTS_TO_PROP and prev_frame.gt_boxes.numel() != 0:
                # match一下，只用检测到的gt的joint
                prev_box_np = prev_filtered_box.numpy()[:, :4]
                gt_box_np = prev_frame.gt_boxes.numpy()[:, :4]
                prev_to_gt_iou = tf_iou(prev_box_np, gt_box_np, )
                
                # prev_box_scores = gt_scores[matched_gt_ind.astype(np.uint8)]  # 从这里下来的score如果是一个，会是向量而不是scalar
                inds = get_matching_indices((prev_to_gt_iou))
                num_matched_people = len(inds)
                # Overwrite
                prev_frame_joints = torch.zeros([num_matched_people, self.cfg.DATA.num_joints, 2])
                prev_frame_joints_vis = torch.zeros([num_matched_people, self.cfg.DATA.num_joints])
                prev_box_scores = torch.zeros([num_matched_people])
                for i, ind in enumerate(inds):
                    prev, gt = ind
                    prev_frame_joints[i] = prev_frame.gt_joints[gt, :, :2]
                    prev_frame_joints_vis[i] = prev_frame.gt_joints[gt, :, 2]
                    prev_box_scores[i] = prev_filtered_box[prev, 4]
                prev_frame_joints_scores = prev_frame_joints_vis
        
        # preprocess end
        
        if prev_frame_joints.numel() != 0:
            
            # what if no prev_joints
            prev_frame_joints = check_joints_format(prev_frame_joints)
            
            prev_frame_joints = prev_frame_joints.int()
            
            new_joints = torch.zeros(prev_frame_joints.shape)
            # using previous score as new score
            prop_boxes_scores = expand_vector_to_tensor(prev_box_scores).transpose(0, 1)  # tensor numpeople x 1
            
            for people in range(new_joints.shape[0]):
                # debug_print('people', people)
                for joint in range(self.cfg.DATA.num_joints):
                    joint_pos = prev_frame_joints[people, joint, :]  # x,y
                    joint_vis = prev_frame_joints_vis[people, joint]
                    new_joint_vis = joint_pos[0] < self.img_w and joint_pos[1] < self.img_h \
                                    and joint_pos[0] > 0 and joint_pos[1] > 0 and joint_vis > 0
                    if new_joint_vis:
                        # TODO 可能和作者不一样的地方，画面外的joint怎么办，我设定的是两个0，然后自然而然就会被clamp
                        joint_flow = self.flow_to_current[:, joint_pos[1], joint_pos[0]].cpu()
                    else:
                        joint_flow = torch.zeros(2)
                    # TODO 之前new_joints里面没有0，但现在因为引入了gtbox，就可能有（没有标注的关节坐标和vis都是0）
                    # 当然非gtprop的都是1
                    new_joints[people, joint, :] = prev_frame_joints[people, joint, :].float() + joint_flow
            
            debug_print('Proped', new_joints.shape[0], 'boxes', indent = 1)
            # for similarity. no need to expand here cause if only prev_joints has the right dimension
            self.new_joints = new_joints
            self.new_joints_vis = prev_frame_joints_vis
            # calc new bboxes from new joints, dont clamp joint, clamp box
            prop_boxes = joints_to_boxes(self.new_joints, joint_vis = prev_frame_joints_vis,
                                         clamp_size = (self.img_w, self.img_h))
            # add prev_box_scores
            prop_joints_scores = prev_frame_joints_scores.mean(1)
            prop_joints_scores.unsqueeze_(1)
            final_prop_box_scores = prop_boxes_scores * prop_joints_scores
            
            self.prop_boxes = torch.cat([prop_boxes, final_prop_box_scores], 1)
            self.prop_boxes = expand_vector_to_tensor(self.prop_boxes)
            if self.cfg.DEBUG.VIS_PROPED_JOINTS:
                p = PurePosixPath(self.img_path)
                out_dir = os.path.join(self.cfg.PATH.JOINTS_DIR + '_flowproped', p.parts[-2])
                mkdir(out_dir)
                num_people, num_joints, _ = self.new_joints.shape
                joints_to_draw = torch.zeros([2 * num_people, num_joints, 2])
                joint_visible = torch.ones([2 * num_people, num_joints, 1])
                
                for i in range(num_people):
                    # debug_print('drawing flow proped', i, self.joint_prop_bboxes[i], indent = 1)
                    joints_i = self.new_joints[i, ...]  # 16 x 2
                    prev_joints_i = prev_frame_joints[i, ...].float()
                    joints_to_draw[2 * i: 2 * i + 2, :, :] = torch.stack([joints_i, prev_joints_i], 0)
                nise_batch_joints = torch.cat([joints_to_draw, joint_visible], 2)  # 16 x 3
                
                save_single_whole_image_with_joints(
                    im_to_torch(self.original_img).unsqueeze(0),
                    nise_batch_joints,
                    os.path.join(out_dir, self.img_name + ".jpg"),
                    boxes = self.prop_boxes
                )
        else:
            debug_print('Proped 0 boxes', indent = 1)
        self.joints_proped = True
    
    # @log_time('\tNMS……')
    def unify_bbox(self):
        """ - merge bbox; nms; """
        
        def set_empty_unified_box():
            # kakunin! No box at all in this image
            self.unified_boxes = torch.tensor([])
            self.NO_BOXES = True
        
        if self.cfg.DEBUG.NO_NMS:
            self.unified_boxes = self.detected_boxes
            self.boxes_unified = True
            debug_print('Use all detection results in NMS.', indent = 1)
            return
        
        if self.is_first or self.task == 1:
            # if the first or single frame prediction
            all_bboxes = self.detected_boxes
        else:
            if not self.joints_proped or not self.human_detected:
                raise ValueError(
                    'Should run people detection and joints propagation first')
            all_bboxes = torch.cat([self.detected_boxes, self.prop_boxes])  # x 5
        
        if all_bboxes.numel() == 0:
            # no box, set to 0 otherwise error will occur in the next line,
            # in scores = all_bboxes[:, 4]  # vector
            set_empty_unified_box()
        else:
            scores = all_bboxes[:, 4]  # vector
            # debug_print('Before NMS:', scores.shape[0], 'people. ', indent = 1)
            # scores = torch.stack([torch.zeros(num_people), scores], 1)  # num_people x 2
            # cls_all_bboxes = torch.cat([torch.zeros(num_people, 4), all_bboxes[:, :4]], 1)  # num people x 8
            # scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, cls_all_bboxes, 2)
            
            # self.unified_boxes = to_torch(cls_boxes[1])  # 0 for bg, num_nmsed x 5
            # 土法nms
            nms_thres_1 = self.cfg.ALG.UNIFY_NMS_THRES_1
            nms_thres_2 = self.cfg.ALG.UNIFY_NMS_THRES_2
            k_1 = np.where(scores >= nms_thres_1)[0]
            filtered_scores = scores[k_1]
            if filtered_scores.numel() != 0:
                
                boxes = all_bboxes[k_1, :4].numpy()
                order = np.argsort(-filtered_scores)
                keep = []
                if self.cfg.ALG.USE_COCO_IOU_IN_NMS:
                    from pycocotools.mask import iou
                    ious = iou(boxes, boxes, np.zeros(0))
                else:
                    ious = tf_iou(boxes, boxes)
                while order.numel() > 0:
                    i = order[0].item()
                    keep.append(i)
                    ovr = ious[i, order[1:]]
                    inds_to_keep = np.where(ovr <= nms_thres_2)[0]
                    order = order[inds_to_keep + 1]
                
                self.unified_boxes = all_bboxes[k_1][keep]
            self.unified_boxes = expand_vector_to_tensor(self.unified_boxes)
            if self.unified_boxes.numel() == 0:
                set_empty_unified_box()
            # debug_print('After NMS:', self.unified_boxes.shape[0], 'people', indent = 1)
        self.boxes_unified = True
    
    def get_filtered_bboxes_with_thres(self, thres):
        if not self.boxes_unified:
            raise ValueError("Should unify bboxes first")
        if self.NO_BOXES:
            # if no bboxes at all
            return torch.tensor(self.unified_boxes), None
        filtered, valid_score_idx = filter_bbox_with_scores(self.unified_boxes, thres)
        final_valid_idx = valid_score_idx
        return filtered, final_valid_idx
    
    @log_time('\t关节预测……')
    def est_joints(self, joint_detector):
        """detect current image's bboxes' joints pos using people-pose-estimation - """
        
        if not self.boxes_unified and not self.is_first:
            # if first, no unified box because no flow to prop, so dont raise
            raise ValueError('Should unify bboxes first')
        p = PurePosixPath(self.img_path)
        
        out_dir = os.path.join(self.cfg.PATH.JOINTS_DIR + "_single", p.parts[-2])
        mkdir(out_dir)
        torch_img = im_to_torch(self.img_outside_flow).unsqueeze(0)
        self.joints = torch.zeros(self.unified_boxes.shape[0], self.cfg.DATA.num_joints, 2)  # FloatTensor
        self.joints_score = torch.zeros(self.unified_boxes.shape[0], self.cfg.DATA.num_joints)  # FloatTensor
        # if no people boxes, self.joints is tensor([])
        joint_detector.eval()
        for i in range(self.unified_boxes.shape[0]):
            bb = self.unified_boxes[i, :4]  # no score
            human_score = self.unified_boxes[i, 4]
            
            center, scale = box2cs(bb, self.img_ratio)
            rotation = 0
            # from simple/JointDataset.py
            trans = get_affine_transform(center, scale, rotation, self.joint_est_mode_size)
            # In simple_cfg, 0 is h/1 for w. in warpAffine should be (w,h)
            # 哦是因为192是宽256是高
            resized_human_np = cv2.warpAffine(
                self.img_outside_flow,  # hw3
                trans,
                (int(self.joint_est_mode_size[0]), int(self.joint_est_mode_size[1])),
                flags = cv2.INTER_LINEAR)
            if self.cfg.DEBUG.VIS_SINGLE_NO_JOINTS == True:
                cv2.imwrite(os.path.join(out_dir, p.stem + "_nojoints_" + str(i) + '.jpg'), resized_human_np)
            
            # 'images_joint/valid/015860_mpii_single_person/00000001_0.jpg'
            resized_human = im_to_torch(resized_human_np)
            resized_human.unsqueeze_(0)  # make it batch so we can use detector
            joint_hmap = joint_detector(resized_human)
            
            # FLIP 
            if True:  # from function.py/validate
                # this part is ugly, because pytorch has not supported negative index
                flip_pairs = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9]]
                joint_hmap_flipped = np.flip(joint_hmap.cpu().numpy(), 3).copy()
                joint_hmap_flipped = torch.from_numpy(joint_hmap_flipped).cuda()
                output_flipped = joint_detector(joint_hmap_flipped)
                output_flipped = flip_back(output_flipped.cpu().numpy(), flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
                
                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]
                    # output_flipped[:, :, :, 0] = 0
                
                output = (output + output_flipped) * 0.5
            
            # make it batch so we can get right preds
            c_unsqueezed, s_unsqueezed = np.expand_dims(center, 0), np.expand_dims(scale, 0)
            preds, max_val = get_final_preds(
                simple_cfg, joint_hmap.detach().cpu().numpy(),
                c_unsqueezed, s_unsqueezed)  # bs x 16 x 2,  bs x 16 x 1
            
            self.joints[i, :, :] = to_torch(preds).squeeze()
            self.joints_score[i, :] = to_torch(max_val).squeeze() * human_score
            
            if self.cfg.DEBUG.VIS_EST_SINGLE:
                # debug_print(i, indent = 1)
                img_with_joints = get_batch_image_with_joints(torch_img, to_torch(preds), torch.ones(1, 15, 1))
                resized_human_np_with_joints = cv2.warpAffine(
                    img_with_joints,  # hw3
                    trans,
                    (int(self.joint_est_mode_size[0]), int(self.joint_est_mode_size[1])),
                    flags = cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(out_dir, p.stem + "_" + "{:02d}".format(i) + '.jpg'),
                            resized_human_np_with_joints)
        self.joints = expand_vector_to_tensor(self.joints, 3)
        
        self.joints_detected = True
    
    # @log_time('\tID分配……')
    def assign_id(self, Q, get_dist_mat = None) -> None:
        """ input: distance matrix; output: correspondence   """
        """ - Associate ids. question: how to associate using more than two frames?between each 2?- """
        if not self.joints_detected:
            raise ValueError('Should detect joints first')
        self.id_boxes, self.id_idx_in_unified = self.get_filtered_bboxes_with_thres(self.cfg.ALG.ASSIGN_BOX_THRES)
        
        self.people_ids = torch.zeros(self.id_boxes.shape[0]).long()
        
        if not self.id_boxes.numel() == 0:
            
            if self.is_first:
                # if it's the first frame, just assign every, starting from 1
                # no problem when no people detected
                self.people_ids = torch.tensor(range(1, self.id_boxes.shape[0] + 1)).long()
                FrameItem.max_id = self.id_boxes.shape[0]  # not +1
            else:
                # if no boxes no need for matching, and none for the next frame
                prev_frame = Q[-1]
                if not prev_frame.people_ids.numel() == 0:
                    # if there are id in the previous frame
                    if self.cfg.ALG.MATCHING_METRIC == self.cfg.ALG.MATCHING_BOX:
                        # 2TODO: 如果current 没有box，就不会进入这个分支；但是进入这个分支之后，如果current有了，prev没有怎么办？
                        cur_boxes = self.id_boxes.numpy()[:, :4]
                        pre_boxes = prev_frame.id_boxes.numpy()[:, :4]
                        # wat if empty
                        dist_mat = tf_iou(cur_boxes, pre_boxes)
                    else:
                        proped_joints = self.new_joints.squeeze()  # squeeze is not for the first dim
                        proped_ids = prev_frame.human_ids
                        
                        # ─── MATCHING ──────────────────────────────
                        proped_joints = expand_vector_to_tensor(proped_joints,
                                                                3)  # unsqueeze if accidental injury happens
                        assert (proped_joints.shape[0] == len(proped_ids))
                        
                        if len(proped_ids) == 0:
                            # if no people in the previous frame, consecutively
                            self.people_ids = torch.tensor(
                                range(FrameItem.max_id + 1, FrameItem.max_id + self.id_boxes.shape[0] + 1)).long()
                            FrameItem.max_id = FrameItem.max_id + self.id_boxes.shape[0]  # not +1
                        else:
                            id_joints = self.joints[self.id_idx_in_unified, :, :]
                            id_joints = expand_vector_to_tensor(id_joints,
                                                                3)  # in case only one people is in this image
                            dist_mat = get_dist_mat(id_joints, proped_joints)
                    
                    # dist_mat should be np.ndarray
                    if self.cfg.ALG.MATCHING_ALG == self.cfg.ALG.MATCHING_MKRS:
                        indices = get_matching_indices(dist_mat)
                    elif self.cfg.ALG.MATCHING_ALG == self.cfg.ALG.MATCHING_GREEDY:
                        indices = list(zip(*bipartite_matching_greedy(dist_mat)))
                    # debug_print('\t'.join(['(%d, %d) -> %.2f; ID %d' % (
                    #     prev, cur, dist_mat[cur][prev], prev_frame.people_ids[prev])
                    #                        for cur, prev in indices]), indent = 1)
                    for cur, prev in indices:
                        # value = dist_mat[cur][prev]
                        # debug_print('(%d, %d) -> %f' % (cur, prev, value))
                        self.people_ids[cur] = prev_frame.people_ids[prev]
                for i in range(self.people_ids.shape[0]):
                    if self.people_ids[i] == 0:  # unassigned
                        self.people_ids[i] = FrameItem.max_id = FrameItem.max_id + 1
        # debug_print('ID Assigned')
        self.id_assigned = True
    
    # @log_time('\tID分配……')
    def assign_id_using_gt_id(self, get_dist_mat = None) -> None:
        """ input: distance matrix; output: correspondence   """
        if not self.joints_detected:
            raise ValueError('Should detect joints first')
        self.id_boxes, self.id_idx_in_unified = self.get_filtered_bboxes_with_thres(self.cfg.ALG.ASSIGN_BOX_THRES)
        
        self.people_ids = torch.zeros(self.id_boxes.shape[0]).long() - 1  # 剩下的都是-1，就是乱搞了
        inds = []  # init
        if self.gt_id.numel() != 0 and self.id_boxes.numel() != 0:
            # match一下，只用检测到的gt的joint
            id_box_np = self.id_boxes.numpy()[:, :4]
            gt_box_np = self.gt_boxes.numpy()[:, :4]
            prev_to_gt_iou = tf_iou(id_box_np, gt_box_np, )
            inds = get_matching_indices((prev_to_gt_iou))
            # Overwrite
            for i, ind in enumerate(inds):
                prev, gt = ind
                self.people_ids[prev] = self.gt_id[gt]
        else:
            self.people_ids = torch.tensor(range(1, self.id_boxes.shape[0] + 1)).long()
        
        # debug_print('ID Assigned')
        self.id_assigned = True
    
    # @log_time('\tID分配……')
    def assign_id_task_1_2(self, Q, get_dist_mat = None):
        """ input: distance matrix; output: correspondence   """
        """ - Associate ids. question: how to associate using more than two frames?between each 2?- """
        if not self.joints_detected:
            raise ValueError('Should detect joints first')
        self.id_boxes, self.id_idx_in_unified = self.get_filtered_bboxes_with_thres(self.cfg.ALG.ASSIGN_BOX_THRES)
        
        self.people_ids = torch.zeros(self.id_boxes.shape[0]).long()
        
        # debug_print('ID Assigned')
        self.id_assigned = True
    
    def _resize_x(self, x):
        return x * self.ori_img_w / self.img_w
    
    def _resize_y(self, y):
        return y * self.ori_img_h / self.img_h
    
    def _resize_joints(self, joints):
        '''
        
        :param joints: num people x 16 x 2
        :return:
        '''
        joints = expand_vector_to_tensor(joints, 3)
        resized_joints = torch.zeros(joints.shape)
        resized_joints[:, :, 0] = self._resize_x(joints[:, :, 0])
        resized_joints[:, :, 1] = self._resize_y(joints[:, :, 1])
        return resized_joints
    
    @log_time('\t画图……')
    def visualize(self):
        if self.id_assigned is False and self.task != 1:
            raise ValueError('Should assign id first.')
        if self.cfg.DEBUG.VIS_BOX:
            class_boxes = [[]] * 81
            # filter for showing joints
            class_boxes[1] = self.id_boxes
            training_start_time = time.strftime("%H-%M-%S", time.localtime())
            p = PurePosixPath(self.img_path)
            out_dir = os.path.join(self.cfg.PATH.IMAGES_OUT_DIR, p.parts[-2])
            mkdir(out_dir)
            
            vis_utils.vis_one_image_for_pt(
                self.img_outside_flow[:, :, ::-1],  # BGR -> RGB for visualization
                self.img_name,
                out_dir,
                class_boxes,
                None,
                None,
                box_alpha = 0.3,  # opacity
                thresh = self.cfg.DEBUG.VIS_HUMAN_THRES,
                human_ids = self.people_ids,
                kp_thresh = 2,
                ext = 'jpg'
            )
        
        # SHOW JOINTS
        if self.cfg.DEBUG.VIS_JOINTS_FULL:
            if self.joints.numel() == 0 or self.id_idx_in_unified.numel() == 0:
                # Essentially no joints or no joints after filtering
                return
            joints_to_show = self.joints[self.id_idx_in_unified]
            joints_to_show = self._resize_joints(joints_to_show)
            num_people, num_joints, _ = joints_to_show.shape
            
            out_dir = os.path.join(self.cfg.PATH.JOINTS_DIR, p.parts[-2])
            mkdir(out_dir)
            
            joint_visible = torch.ones([num_people, num_joints, 1])
            
            nise_batch_joints = torch.cat([joints_to_show, joint_visible], 2)  # 16 x 3
            
            save_single_whole_image_with_joints(
                im_to_torch(self.original_img).unsqueeze(0),
                nise_batch_joints,
                os.path.join(out_dir, self.img_name + "_withbox.jpg"),
                boxes = self.unified_boxes[self.id_idx_in_unified]
            )
            
            save_single_whole_image_with_joints(
                im_to_torch(self.original_img).unsqueeze(0),
                nise_batch_joints,
                os.path.join(out_dir, self.img_name + "_nobox.jpg"),
                boxes = None
            )
            for i in range(num_people):
                print(i)
                nise_batch_joints = torch.cat([joints_to_show[i, ...], joint_visible[i, ...]], 1)  # 16 x 3
                ID = self.people_ids[i].item() if self.people_ids[i].item() != 0 else i
                save_single_whole_image_with_joints(
                    im_to_torch(self.original_img).unsqueeze(0),
                    nise_batch_joints.unsqueeze(0),
                    os.path.join(out_dir, self.img_name + "_id_" + "{:02d}".format(ID) + ".jpg"),
                    boxes = self.id_boxes[i].unsqueeze(0), human_ids = self.people_ids[i].unsqueeze(0)
                )
    
    # @log_time('\tTo dict......')
    def to_dict(self):
        '''
        Output string for json dump
        Here the size is flow_input_size = (1024, 576), need to map back to original coord.

        :return:
        '''
        if self.task == 1 or self.task == 2:
            # output all joints
            d = {
                'image': [
                    {
                        'name': self.img_path.replace(self.cfg.PATH.POSETRACK_ROOT, ''),
                    }
                ],
                'annorect': [  # i for people
                    {
                        'x1': [0],
                        'x2': [0],
                        'y1': [0],
                        'y2': [0],
                        'score': [-1],
                        'track_id': [0],
                        'annopoints': [
                            {
                                'point': [
                                    {  # j for joints
                                        'id': [j],
                                        'x': [self.joints[i, j, 0].item() * self.ori_img_w / self.img_w],
                                        'y': [self.joints[i, j, 1].item() * self.ori_img_h / self.img_h],
                                        'score': [self.joints_score[i, j].item()]
                                    } for j in range(self.cfg.DATA.num_joints)
                                ]
                            }
                        ]
                    } for i in range(self.unified_boxes.shape[0])
                
                ]
                # 'imgnum'
            }
        
        else:
            num_person = self.people_ids.shape[0]
            d = {
                'image': [
                    {
                        'name': self.img_path.replace(self.cfg.PATH.POSETRACK_ROOT, ''),
                    }
                ],
                'annorect': [  # i for people
                    {
                        'x1': [0],
                        'x2': [0],
                        'y1': [0],
                        'y2': [0],
                        'score': [self.id_boxes[i, 4].item()],
                        'track_id': [self.people_ids[i].item()],
                        'annopoints': [
                            {
                                'point': [
                                    {  # j for joints
                                        'id': [j],
                                        'x': [self.joints[i, j, 0].item() * self.ori_img_w / self.img_w],
                                        'y': [self.joints[i, j, 1].item() * self.ori_img_h / self.img_h],
                                        'score': [self.joints_score[i, j].item()]
                                    } for j in range(self.cfg.DATA.num_joints) if
                                    self.joints_score[i, j] >= self.cfg.ALG.OUTPUT_JOINT_THRES
                                    # don t report those small with small scores
                                ]
                            }
                        ]
                    } for i in range(num_person)
                
                ]
                # 'imgnum'
            }
        
        self.pt_eval_dict = d
        return d
    
    def detect_results(self):
        return self.detected_boxes.tolist()
    
    def unfied_result(self):
        return self.unified_boxes
    
    def flow_result(self):
        return self.flow_to_current
    
    def det_est_result(self):
        '''
        Only used in task 1
        :return:
        '''
        if self.joints_score.numel() != 0:
            final_jt = torch.cat([self.joints, self.joints_score.unsqueeze(2)], 2)
        else:
            final_jt = torch.tensor([])
        return final_jt
    
    def new_joints_result(self):
        return self.new_joints
