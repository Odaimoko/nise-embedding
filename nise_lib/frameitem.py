import sys

sys.path.append('..')
from munkres import Munkres
from pathlib import PurePosixPath
import scipy
import cv2
import time
from nise_lib.nise_functions import *
from nise_lib.nise_debugging_func import *
from nise_lib.nise_config import cfg as nise_cfg
from nise_utils.simple_vis import save_batch_image_with_joints, get_batch_image_with_joints
import tron_lib.utils.vis as vis_utils
from tron_lib.core.test_for_pt import im_detect_all, box_results_with_nms_and_limit
from simple_lib.core.inference import get_final_preds
from simple_lib.core.config import config as simple_cfg
from nise_utils.transforms import get_affine_transform


class FrameItem:
    mkrs = Munkres()
    max_id = 0
    '''
        Item in Deque, storing image (for flow generation), joints(for propagation),
         trackids.
         One image or a batch of images?
    '''
    
    def __init__(self, img_path, task = 2, is_first = False):
        '''

        :param is_first: is this frame the first of the sequence
        '''
        self.task = task  # 1 for single-frame, 2 for multi-frame and tracking
        self.is_first = is_first
        self.img_path = img_path
        self.img_name = PurePosixPath(img_path).stem
        # tensor with size C x H x W,  detector needs BGR so here use bgr.
        # flow uses rgb
        self.original_img = cv2.imread(self.img_path)  # with original size? YES
        self.ori_img_h, self.ori_img_w, _ = self.original_img.shape
        
        """ - Normalize person crop to some size to fit flow input
        
        # data[0] torch.Size([8, 3, 2, 384, 1024]) ，bs x channels x num_images, h, w
        # target torch.Size([8, 2, 384, 1024]) maybe offsets
        
        """
        if nise_cfg.TEST.USE_GT_VALID_BOX and self.task == 1:
            # dont change image size ok?
            self.bgr_img = self.original_img
        else:
            self.bgr_img = scipy.misc.imresize(  # h and w is reversed
                self.original_img,
                (nise_cfg.DATA.flow_input_size[1], nise_cfg.DATA.flow_input_size[0])
            )
        
        # only used in computation of center and scale
        self.img_h, self.img_w, _ = self.bgr_img.shape
        # NG, should be model size
        # self.img_ratio = self.img_w * 1.0 / self.img_h
        # 依然是W/H， 回忆model第一个192是宽。
        self.img_ratio = simple_cfg.MODEL.IMAGE_SIZE[0] / simple_cfg.MODEL.IMAGE_SIZE[1]
        # self.rgb_img = load_image(self.img_name)  # with original size? YES   ~~no, resize to some fixed size~~
        
        # tensor num_person x num_joints x 2
        # should be LongTensor since we are going to use them as index. same length with unified_bbox
        self.joints = torch.tensor([])
        # tensor
        self.new_joints = torch.tensor([])  # for similarity
        # tensor, in the resized img for flow.
        self.flow_to_current = None  # 2, h, w
        
        # tensors, coord in the original image
        self.detected_bboxes = torch.tensor([])  # num_people x 5, with score
        self.joint_prop_bboxes = torch.tensor([])
        self.unified_bboxes = torch.tensor([])
        self.id_bboxes = torch.tensor([])  # those have ids. should have the same length with human_ids
        self.id_idx_in_unified = None
        
        self.human_ids = None
        
        # ─── FLAGS FOR CORRENT ORDER ─────────────────────────────────────
        self.human_detected = False
        self.flow_calculated = False
        self.joints_proped = False
        self.bboxes_unified = False
        self.joints_detected = False
        self.id_assigned = False
        
        self.NO_BBOXES = False
    
    def detect_human(self, detector, gt_joints = None):
        '''
        """ - detect person: do we use all the results? - """

        :param detector:
        :return: human is represented as tensor of size num_people x 4. The result is NMSed.
        '''
        
        if nise_cfg.TEST.USE_GT_VALID_BOX and gt_joints is not None:
            if gt_joints.numel() == 0:
                self.detected_bboxes = torch.tensor([])
            else:
                gt_bbox = joints_to_bboxes(gt_joints[:, :, :2], gt_joints[:, :, 2], (self.img_w, self.img_h))
                gt_bbox = expand_vector_to_tensor(gt_bbox)
                gt_scores = torch.ones([gt_bbox.shape[0]])
                gt_scores.unsqueeze_(1)
                self.detected_bboxes = torch.cat([gt_bbox, gt_scores], 1)
        
        else:
            # no problem for RGB BGR, since the example from detectron use this in this way
            # self.img is a tensor with size C x H x W, but detector needs H x W x C and BGR
            cls_boxes = im_detect_all(detector, self.bgr_img)
            human_bboxes = torch.from_numpy(cls_boxes[1])  # person is the first class of coco， 0 for background
            if nise_cfg.ALG.FILTER_HUMAN_WHEN_DETECT:
                # Dont use all, only high confidence
                self.detected_bboxes, idx = filter_bbox_with_scores(human_bboxes)
            else:
                self.detected_bboxes = human_bboxes
            # 有可能出现没检测到的情况，大部分不可能，但是工程起见。
            # TODO 比如 bonn_mpii_train_5sec\00098_mpii 最前面几个是全黑所以检测不到……emmmm怎么办呢
            self.detected_bboxes = expand_vector_to_tensor(self.detected_bboxes)
        self.human_detected = True
    
    def gen_flow(self, flow_model, prev_frame_img = None):
        # No precedent functions, root
        if self.is_first:  # if this is first frame
            return
        """ - Now we have some images Q and current img, calculate FLow using previous and current one - """
        resized_img = self.bgr_img[:, :, ::-1]  # to rgb
        resized_prev = prev_frame_img[:, :, ::-1]
        resized_img = im_to_torch(resized_img)
        resized_prev = im_to_torch(resized_prev)
        ti = torch.stack([resized_prev, resized_img], 1)
        self.flow_to_current = pred_flow(ti, flow_model)
        self.flow_calculated = True
    
    def joint_prop(self, Q):
        '''

        :param prev_frame_joints: 2 x num_people x num_joints
        :return:
        '''
        if self.is_first: return
        if not self.flow_calculated:
            raise ValueError('Flow not calculated yet.')
        """ - choose prev joints pos, add flow to determine pos in the next frame; generate new bbox - """
        
        def set_empty_joint():
            self.joints_proped = True
        
        # preprocess
        prev_frame = Q[-1]
        if not prev_frame.joints.numel() == 0:
            # if there  are joints to propagate
            if nise_cfg.ALG.JOINT_PROP_WITH_FILTERED_HUMAN:
                prev_filtered_box, filtered_idx = prev_frame.get_filtered_bboxes()
                if prev_filtered_box.numel() == 0:
                    set_empty_joint()
                    return
                
                prev_frame_joints = prev_frame.joints[filtered_idx]
                scores = prev_frame.id_bboxes[:, -1]
            else:  # use all prev joints rather than filtered, or should we?
                prev_frame_joints = prev_frame.joints
                scores = prev_frame.unified_bboxes[:, -1]
            prev_frame_joints = expand_vector_to_tensor(prev_frame_joints, 3)
            
            if not ((prev_frame_joints.shape[2] == nise_cfg.DATA.num_joints and prev_frame_joints.shape[0] == 2) or (
                    prev_frame_joints.shape[2] == 2 and prev_frame_joints.shape[1] == nise_cfg.DATA.num_joints)):
                raise ValueError(
                    'Size not matched, current size ' + str(prev_frame_joints.shape))
            if prev_frame_joints.shape[2] == nise_cfg.DATA.num_joints:
                # :param prev_frame_joints: 2x num_people x num_joints. to num_people x num_joints x 2
                prev_frame_joints = torch.transpose(prev_frame_joints, 0, 1)
                prev_frame_joints = torch.transpose(prev_frame_joints, 1, 2)
            
            prev_frame_joints = prev_frame_joints.int()
            
            # if nise_cfg.ALG.JOINT_PROP_WITH_FILTERED_HUMAN:
            #     # Dont use all, only high confidence
            #     valid_scores_idx = torch.nonzero(scores >= nise_cfg.ALG._HUMAN_THRES).squeeze_()
            #     joint_prop_bboxes_scores = scores[valid_scores_idx]  # vector
            #
            #     new_joints = torch.zeros([len(joint_prop_bboxes_scores), nise_cfg.DATA.num_joints, 2])
            #
            #     # the ith in new_joints is the valid_scores_idx[i] th in prev_frame_joints
            #     for person in range(new_joints.shape[0]):
            #         # if we dont use item, o_p will be a tensor and
            #         # use tensor to index prev_frame_joints resulting in size [1,2] of joint_pos
            #         original_person = valid_scores_idx[person].item()
            #         for joint in range(nise_cfg.DATA.num_joints):
            #             joint_pos = prev_frame_joints[original_person, joint, :]  # x,y
            #             # if joint pos is inside the image, use the flow; otherwise set to 0.
            #             if joint_pos[0] < self.img_w and joint_pos[1] < self.img_h:
            #                 # this is cuda so turn it into cpu
            #                 joint_flow = self.flow_to_current[:, joint_pos[1], joint_pos[0]].cpu()
            #             else:
            #                 joint_flow = torch.zeros(2)
            #             new_joints[person, joint, :] = prev_frame_joints[original_person, joint, :].float() + joint_flow
            # else:
            new_joints = torch.zeros(prev_frame_joints.shape)
            
            joint_prop_bboxes_scores = scores  # vector
            
            for person in range(new_joints.shape[0]):
                for joint in range(nise_cfg.DATA.num_joints):
                    joint_pos = prev_frame_joints[person, joint, :]  # x,y
                    if joint_pos[0] < self.img_w and joint_pos[1] < self.img_h \
                            and joint_pos[0] >= 0 and joint_pos[1] >= 0:
                        joint_flow = self.flow_to_current[:, joint_pos[1], joint_pos[0]].cpu()
                    else:
                        joint_flow = torch.zeros(2)
                    new_joints[person, joint, :] = prev_frame_joints[person, joint, :].float() + joint_flow
            
            # for similarity. no need to expand here cause if only prev_joints has the right dimension
            self.new_joints = new_joints
            # calc new bboxes from new joints
            joint_prop_bboxes = joints_to_bboxes(self.new_joints, nise_cfg.DATA.flow_input_size)
            # add scores
            joint_prop_bboxes_scores.unsqueeze_(1)
            self.joint_prop_bboxes = torch.cat([joint_prop_bboxes, joint_prop_bboxes_scores], 1)
            self.joint_prop_bboxes = expand_vector_to_tensor(self.joint_prop_bboxes)
        
        self.joints_proped = True
    
    def unify_bbox(self):
        """ - merge bbox; nms; """
        
        def set_empty_unified_bbox():
            
            self.unified_bboxes = torch.tensor([])
            # kakunin! No box at all in this image
            self.NO_BBOXES = True
        
        if self.is_first or self.task == 1:
            # if the first or single frame prediction
            all_bboxes = self.detected_bboxes
        else:
            if not self.joints_proped or not self.human_detected:
                raise ValueError(
                    'Should run human detection and joints propagation first')
            num_classes = 2  # bg and human
            all_bboxes = torch.cat([self.detected_bboxes, self.joint_prop_bboxes])  # x 5
        
        num_people = all_bboxes.shape[0]
        
        if all_bboxes.numel() == 0:
            # no box, set to 0 otherwise error will occur in the next line,
            # in scores = all_bboxes[:, 4]  # vector
            set_empty_unified_bbox()
        else:
            scores = all_bboxes[:, 4]  # vector
            scores = torch.stack([torch.zeros(num_people), scores], 1)  # num_people x 2
            cls_all_bboxes = torch.cat([torch.zeros(num_people, 4), all_bboxes[:, :4]], 1)  # num people x 8
            scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, cls_all_bboxes, 2)
            self.unified_bboxes = to_torch(cls_boxes[1])  # 0 for bg, num_nmsed x 5
            # Althought we dont filter out boxes with low score, we leave out those with small areas
            # QQ 这里这样正确吗，需不需要filter？thres参数又是怎么样？
            if nise_cfg.ALG.FILTER_BBOX_WITH_SMALL_AREA:
                self.unified_bboxes, _ = filter_bbox_with_area(self.unified_bboxes)
            self.unified_bboxes = expand_vector_to_tensor(self.unified_bboxes)
            if self.unified_bboxes.numel() == 0:
                set_empty_unified_bbox()
        self.bboxes_unified = True
    
    def get_filtered_bboxes(self):
        if not self.bboxes_unified:
            raise ValueError("Should unify bboxes first")
        if self.NO_BBOXES:
            # if no bboxes at all
            return torch.tensor(self.unified_bboxes), None
        filtered, valid_score_idx = filter_bbox_with_scores(self.unified_bboxes)
        final_valid_idx = valid_score_idx
        return filtered, final_valid_idx
    
    def est_joints(self, joint_detector):
        """detect current image's bboxes' joints pos using human-pose-estimation - """
        
        p = PurePosixPath(self.img_path)
        
        out_dir = os.path.join(nise_cfg.PATH.JOINTS_DIR + "_single", p.parts[-2] + '_task_' + str(self.task))
        mkdir(out_dir)
        torch_img = im_to_torch(self.bgr_img).unsqueeze(0)
        if not self.bboxes_unified and not self.is_first:
            # if first, no unified box because no flow to prop, so dont raise
            raise ValueError('Should unify bboxes first')
        self.joints = torch.zeros(self.unified_bboxes.shape[0], nise_cfg.DATA.num_joints, 2)  # FloatTensor
        # if no human boxes, self.joints is tensor([])
        joint_detector.eval()
        for i in range(self.unified_bboxes.shape[0]):
            bb = self.unified_bboxes[i, :4]  # no score
            # debug_print(i, bb)
            
            center, scale = box2cs(bb, self.img_ratio)  # TODO: is this right?
            # scale = scale / 2
            rotation = 0
            # from simple/JointDataset.py
            trans = get_affine_transform(center, scale, rotation, simple_cfg.MODEL.IMAGE_SIZE)
            # In simple_cfg, 0 is h/1 for w. in warpAffine should be (w,h)
            # but when visualization, it seems that the reverse way is correct??
            # 哦是因为192是宽256是高
            resized_human_np = cv2.warpAffine(
                self.bgr_img,  # hw3
                trans,
                (int(simple_cfg.MODEL.IMAGE_SIZE[0]), int(simple_cfg.MODEL.IMAGE_SIZE[1])),
                flags = cv2.INTER_LINEAR)
            # 'images_joint/valid/015860_mpii_single_person/00000001_0.jpg'
            cv2.imwrite(os.path.join(out_dir, p.stem + "_" + str(i) + '.jpg'), resized_human_np)
            resized_human = im_to_torch(resized_human_np)
            resized_human.unsqueeze_(0)  # make it batch so we can use detector
            joint_hmap = joint_detector(resized_human)
            # scale = np.ones([2])  # the scale above is like a 大傻逼
            # make it batch so we can get right preds
            c_unsqueezed, s_unsqueezed = np.expand_dims(center, 0), np.expand_dims(scale, 0)
            preds, _ = get_final_preds(
                simple_cfg, joint_hmap.detach().cpu().numpy(),
                c_unsqueezed, s_unsqueezed)  # bs x 16 x 2,  bs x 16 x 1
            # NG ! NG !
            self.joints[i, :, :] = to_torch(preds).squeeze()
            
            img_with_joints = get_batch_image_with_joints(torch_img, to_torch(preds), torch.ones(1, 15, 1))
            resized_human_np_with_joints = cv2.warpAffine(
                img_with_joints,  # hw3
                trans,
                (int(simple_cfg.MODEL.IMAGE_SIZE[0]), int(simple_cfg.MODEL.IMAGE_SIZE[1])),
                flags = cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(out_dir, p.stem + "_" + "{:02d}".format(i) + '.jpg'),
                        resized_human_np_with_joints)
        self.joints = expand_vector_to_tensor(self.joints, 3)
        
        self.joints_detected = True
    
    def assign_id(self, Q, get_dist_mat = None):
        """ input: distance matrix; output: correspondence   """
        """ - Associate ids. question: how to associate using more than two frames?between each 2?- """
        if not self.joints_detected:
            raise ValueError('Should detect joints first')
        if nise_cfg.ALG.ASSGIN_ID_TO_FILTERED_BOX:
            self.id_bboxes, self.id_idx_in_unified = self.get_filtered_bboxes()
            self.id_bboxes = expand_vector_to_tensor(self.id_bboxes)  # in case only one person ,then we have 5 index
            self.human_ids = torch.zeros(self.id_bboxes.shape[0]).long()
        else:
            self.id_bboxes = self.unified_bboxes
            self.id_bboxes = expand_vector_to_tensor(self.id_bboxes)
            self.id_idx_in_unified = torch.tensor(range(self.unified_bboxes.shape[0])).long()
        
        if self.is_first:
            # if it's the first frame, just assign every, starting from 1
            # no problem when no people detected
            self.human_ids = torch.tensor(range(1, self.id_bboxes.shape[0] + 1)).long()
            FrameItem.max_id = self.id_bboxes.shape[0]  # not +1
        elif not self.NO_BBOXES:
            # if no boxes no need for matching, and none for the next frame
            
            if get_dist_mat is None: raise NotImplementedError('Should pass a matrix function function in')
            
            # proped from prev frame. since we want prev ids, we should get filter ones.
            if nise_cfg.ALG.JOINT_PROP_WITH_FILTERED_HUMAN:  # if filtered when proped, no need to do it here
                prev_joints = self.new_joints.squeeze()  # squeeze is not for the first dim
            else:
                prev_boxes_filtered, prev_boxes_idx = Q[-1].get_filtered_bboxes()
                prev_joints = self.new_joints[prev_boxes_idx].squeeze()  # squeeze is not for the first dim
            # ─── MATCHING ──────────────────────────────
            prev_joints = expand_vector_to_tensor(prev_joints, 3)  # unsqueeze if accidental injury happens
            prev_ids = Q[-1].human_ids
            assert (prev_joints.shape[0] == len(prev_ids))
            
            if len(prev_ids) == 0:
                # if no person in the previous frame, consecutively
                self.human_ids = torch.tensor(
                    range(FrameItem.max_id + 1, FrameItem.max_id + self.id_bboxes.shape[0] + 1)).long()
                FrameItem.max_id = FrameItem.max_id + self.id_bboxes.shape[0]  # not +1
            else:
                id_joints = self.joints[self.id_idx_in_unified, :, :]
                id_joints = expand_vector_to_tensor(id_joints, 3)  # in case only one person is in this image
                dist_mat = get_dist_mat(id_joints, prev_joints)
                # to use munkres package, we need int. munkres minimize cost, so use negative version
                scaled_distance_matrix = -nise_cfg.ALG._OKS_MULTIPLIER * dist_mat
                # but if converted to numpy, will have precision problem
                scaled_distance_matrix = scaled_distance_matrix.numpy()
                mask = (scaled_distance_matrix <= -1e-9).astype(np.float32)
                scaled_distance_matrix *= mask
                indexes = FrameItem.mkrs.compute(scaled_distance_matrix.tolist())
                # print_matrix(scaled_distance_matrix, msg = 'Maximize total cost...')
                for cur, prev in indexes:
                    # value = dist_mat[cur][prev]
                    # debug_print('(%d, %d) -> %f' % (cur, prev, value))
                    self.human_ids[cur] = prev_ids[prev]
                for i in range(self.human_ids.shape[0]):
                    if self.human_ids[i] == 0:  # unassigned
                        self.human_ids[i] = FrameItem.max_id + 1
                        FrameItem.max_id += 1
        debug_print('ID Assigned')
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
    
    def visualize(self, dataset):
        if self.id_assigned is False and self.task != 1:
            raise ValueError('Should assign id first.')
        class_boxes = [[]] * 81
        # filter for showing joints
        class_boxes[1] = self.id_bboxes
        training_start_time = time.strftime("%H-%M-%S", time.localtime())
        p = PurePosixPath(self.img_path)
        out_dir = os.path.join(nise_cfg.PATH.IMAGES_OUT_DIR, p.parts[-2] + '_task_' + str(self.task))
        mkdir(out_dir)
        
        vis_utils.vis_one_image_for_pt(
            self.bgr_img[:, :, ::-1],  # BGR -> RGB for visualization
            self.img_name,
            out_dir,
            class_boxes,
            None,
            None,
            dataset = dataset,
            box_alpha = 0.3,  # opacity
            show_class = True,
            thresh = nise_cfg.DEBUG.VIS_HUMAN_THRES,
            human_ids = self.human_ids,
            kp_thresh = 2,
            ext = 'jpg'
        )
        
        # SHOW JOINTS
        if self.joints.numel() == 0:
            # No joints to draw
            return
        joints_to_show = self.joints[self.id_idx_in_unified]
        joints_to_show = self._resize_joints(joints_to_show)
        num_people, num_joints, _ = joints_to_show.shape
        
        out_dir = os.path.join(nise_cfg.PATH.JOINTS_DIR, p.parts[-2] + '_task_' + str(self.task))
        mkdir(out_dir)
        for i in range(num_people):
            joints_i = joints_to_show[i, ...]  # 16 x 2
            joint_visible = torch.ones([num_joints, 1])
            nise_batch_joints = torch.cat([joints_i, joint_visible], 1)  # 16 x 3
            
            save_batch_image_with_joints(
                im_to_torch(self.original_img).unsqueeze(0),
                nise_batch_joints.unsqueeze(0),
                joint_visible.reshape([1, -1]),
                os.path.join(out_dir, self.img_name + "_id_" + "{:02d}".format(
                    self.human_ids[i].item()) + ".jpg"),
                nrow = 2
            )
    
    def to_dict(self):
        '''
        Output string for json dump
        Here the size is flow_input_size = (1024, 576), need to map back to original coord.

        :return:
        '''
        if self.task == 1:
            
            d = {
                'image': [
                    {
                        'name': self.img_path,
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
                                        'score': [1]  # what score????
                                    } for j in range(nise_cfg.DATA.num_joints)
                                ]
                            }
                        ]
                    } for i in range(self.unified_bboxes.shape[0])
                
                ]
                # 'imgnum'
            }
        
        else:
            d = {
                'image': [
                    {
                        'name': self.img_path,
                    }
                ],
                'annorect': [  # i for people
                    {
                        'score': [self.id_bboxes[i, 4].item()],
                        'track_id': [self.human_ids[i].item()],
                        'x1': [0],
                        'x2': [0],
                        'y1': [0],
                        'y2': [0],
                        'annopoints': [
                            {
                                'point': [
                                    {  # j for joints
                                        'id': [j],
                                        'x': [self.joints[i, j, 0].item() * self.ori_img_w / self.img_w],
                                        'y': [self.joints[i, j, 1].item() * self.ori_img_h / self.img_h],
                                        'score': [-1]  # what score????
                                    } for j in range(nise_cfg.DATA.num_joints)
                                ]
                            }
                        ]
                    } for i in range(self.human_ids.shape[0])
                
                ]
                # 'imgnum'
            }
        
        return d
