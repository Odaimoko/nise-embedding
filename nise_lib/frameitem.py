import sys
sys.path.append('..')
from munkres import Munkres
from pathlib import PurePosixPath
import cv2
import time
from nise_functions import *
from nise_debugging_func import *
from nise_config import cfg as nise_cfg
from nise_utils.simple_vis import save_batch_image_with_joints
import nise_lib._init_paths
import tron_lib.utils.vis as vis_utils
from tron_lib.core.test_for_pt import im_detect_all, box_results_with_nms_and_limit
from simple_lib.core.inference import get_final_preds
# from simple_lib.core.config import config as simple_cfg

class FrameItem:
    mkrs = Munkres()
    max_id = 0
    '''
        Item in Deque, storing image (for flow generation), joints(for propagation),
         trackids.
         One image or a batch of images?
    '''
    
    def __init__(self, img_path, is_first = False):
        '''

        :param is_first: is this frame the first of the sequence
        '''
        self.is_first = is_first
        self.img_path = img_path
        self.img_name = PurePosixPath(img_path).stem
        # tensor with size C x H x W,  detector needs BGR so here use bgr.
        # flow uses rgb
        self.original_img = cv2.imread(self.img_path)  # with original size? YES
        self.bgr_img = scipy.misc.imresize(  # h and w is reversed
            self.original_img,
            (nise_cfg.DATA.flow_input_size[1], nise_cfg.DATA.flow_input_size[0])
        )
        # only used in computation of center and scale
        self.img_h, self.img_w, _ = self.bgr_img.shape
        self.img_ratio = self.img_w * 1.0 / self.img_h
        # self.rgb_img = load_image(self.img_name)  # with original size? YES   ~~no, resize to some fixed size~~
        
        # tensor num_person x num_joints x 2
        # should be LongTensor since we are going to use them as index. same length with unified_bbox
        self.joints = None
        # tensor
        self.new_joints = None  # for similarity
        # tensor, in the resized img for flow.
        self.flow_to_current = None  # 2, h, w
        
        # tensors, coord in the original image
        self.detected_bboxes = None  # num_people x 5, with score
        self.joint_prop_bboxes = None
        self.unified_bboxes = None
        self.id_bboxes = None  # those have ids. should have the same length with human_ids
        self.id_idx_in_unified = None
        
        self.human_ids = None
        
        # ─── FLAGS FOR CORRENT ORDER ─────────────────────────────────────
        self.human_detected = False
        self.flow_calculated = False
        self.joints_proped = False
        self.bboxes_unified = False
        self.joints_detected = False
        self.id_assigned = False
    
    def detect_human(self, detector):
        '''

        :param detector:
        :return: human is represented as tensor of size num_people x 4. The result is NMSed.
        '''
        # no problem for RGB BGR, since the example from detectron use this in this way
        # self.img is a tensor with size C x H x W, but detector needs H x W x C and BGR
        cls_boxes = im_detect_all(detector, self.bgr_img)
        human_bboxes = torch.from_numpy(cls_boxes[1])  # person is the first class of coco， 0 for background
        if nise_cfg.ALG.FILTER_HUMAN_WHEN_DETECT:
            # Dont use all, only high confidence
            self.detected_bboxes, idx = filter_bbox_with_scores(human_bboxes)
        else:
            self.detected_bboxes = human_bboxes
        self.human_detected = True
    
    def gen_flow(self,flow_model, prev_frame_img = None):
        # No precedent functions, root
        if self.is_first:  # if this is first frame
            return
        resized_img = self.bgr_img[:, :, ::-1]  # to rgb
        resized_prev = prev_frame_img[:, :, ::-1]
        resized_img = im_to_torch(resized_img)
        resized_prev = im_to_torch(resized_prev)
        resized_img = resize(resized_img, *nise_cfg.DATA.flow_input_size)
        resized_prev = resize(
            resized_prev, *nise_cfg.DATA.flow_input_size)
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
        
        # preprocess
        prev_frame = Q[-1]
        prev_frame_joints = prev_frame.joints
        if not ((prev_frame_joints.shape[2] == nise_cfg.DATA.num_joints and prev_frame_joints.shape[0] == 2) or (
                prev_frame_joints.shape[2] == 2 and prev_frame_joints.shape[1] == nise_cfg.DATA.num_joints)):
            raise ValueError(
                'Size not matched, current size ' + str(prev_frame_joints.shape))
        if prev_frame_joints.shape[2] == nise_cfg.DATA.num_joints:
            # :param prev_frame_joints: 2x num_people x num_joints. to num_people x num_joints x 2
            prev_frame_joints = torch.transpose(prev_frame_joints, 0, 1)
            prev_frame_joints = torch.transpose(prev_frame_joints, 1, 2)
        
        prev_frame_joints = prev_frame_joints.int()
        
        scores = prev_frame.unified_bboxes[:, -1]
        if nise_cfg.ALG.JOINT_PROP_WITH_FILTERED_HUMAN:
            # Dont use all, only high confidence
            valid_scores_idx = torch.nonzero(scores >= nise_cfg.ALG.HUMAN_THRES).squeeze_()
            joint_prop_bboxes_scores = scores[valid_scores_idx]  # vector
            
            new_joints = torch.zeros([len(joint_prop_bboxes_scores), nise_cfg.DATA.num_joints, 2])
            
            # the ith in new_joints is the valid_scores_idx[i] th in prev_frame_joints
            for person in range(new_joints.shape[0]):
                # if we dont use item, o_p will be a tensor and
                # use tensor to index prev_frame_joints resulting in size [1,2] of joint_pos
                original_person = valid_scores_idx[person].item()
                for joint in range(nise_cfg.DATA.num_joints):
                    joint_pos = prev_frame_joints[original_person, joint, :]  # x,y
                    # if joint pos is inside the image, use the flow; otherwise set to 0.
                    if joint_pos[0] < self.img_w and joint_pos[1] < self.img_h:
                        # this is cuda so turn it into cpu
                        joint_flow = self.flow_to_current[:, joint_pos[1], joint_pos[0]].cpu()
                    else:
                        joint_flow = torch.zeros(2)
                    new_joints[person, joint, :] = prev_frame_joints[original_person, joint, :].float() + joint_flow
        else:
            new_joints = torch.zeros(prev_frame_joints.shape)
            
            joint_prop_bboxes_scores = scores  # vector
            
            for person in range(new_joints.shape[0]):
                for joint in range(nise_cfg.DATA.num_joints):
                    joint_pos = prev_frame_joints[person, joint, :]  # x,y
                    if joint_pos[0] < self.img_w and joint_pos[1] < self.img_h:
                        joint_flow = self.flow_to_current[:, joint_pos[1], joint_pos[0]].cpu()
                    else:
                        joint_flow = torch.zeros(2)
                    new_joints[person, joint, :] = prev_frame_joints[person, joint, :].float() + joint_flow
        
        # for similarity
        self.new_joints = new_joints
        # calc new bboxes from new joints
        joint_prop_bboxes = joints_to_bboxes(self.new_joints)
        # add scores
        joint_prop_bboxes_scores.unsqueeze_(1)
        self.joint_prop_bboxes = torch.cat([joint_prop_bboxes, joint_prop_bboxes_scores], 1)
        self.joints_proped = True
    
    def unify_bbox(self):
        if self.is_first:
            all_bboxes = self.detected_bboxes
        else:
            if not self.joints_proped or not self.human_detected:
                raise ValueError(
                    'Should run human detection and joints propagation first')
            num_classes = 2  # bg and human
            all_bboxes = torch.cat([self.detected_bboxes, self.joint_prop_bboxes])  # x 5
        num_people = all_bboxes.shape[0]
        
        scores = all_bboxes[:, 4]  # vector
        scores = torch.stack([torch.zeros(num_people), scores], 1)  # num_people x 2
        cls_all_bboxes = torch.cat([torch.zeros(num_people, 4), all_bboxes[:, :4]], 1)  # num people x 8
        scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, cls_all_bboxes, 2)
        self.unified_bboxes = to_torch(cls_boxes[1])  # 0 for bg, num_nmsed x 5
        self.bboxes_unified = True
    
    def get_filtered_bboxes(self):
        if not self.bboxes_unified:
            raise ValueError("Should unify bboxes first")
        filtered = filter_bbox_with_scores(self.unified_bboxes)
        return filtered
    
    def assign_id(self, Q, get_dist_mat = None):
        """ input: distance matrix; output: correspondence   """
        if not self.joints_detected:
            raise ValueError('Should detect joints first')
        if nise_cfg.ALG.ASSGIN_ID_TO_FILTERED_BOX:
            self.id_bboxes, self.id_idx_in_unified = self.get_filtered_bboxes()
            self.human_ids = torch.zeros(self.id_bboxes.shape[0]).long()
        else:
            self.id_bboxes = self.unified_bboxes
            self.id_idx_in_unified = torch.tensor(range(self.unified_bboxes.shape[0])).long()
        
        if self.is_first:
            # if it's the first frame, just assign every, starting from 1
            self.human_ids = torch.tensor(range(1, self.id_bboxes.shape[0] + 1)).long()
            FrameItem.max_id = self.id_bboxes.shape[0]  # not +1
        
        else:
            if get_dist_mat is None: raise NotImplementedError('Should pass a matrix function function in')
            # proped from prev frame. since we want prev ids, we should get filter ones.
            prev_boxes_filtered, prev_boxes_idx = Q[-1].get_filtered_bboxes()
            prev_joints = self.new_joints[prev_boxes_idx].squeeze()
            prev_ids = Q[-1].human_ids
            assert (prev_joints.shape[0] == len(prev_ids))
            dist_mat = get_dist_mat(self.joints[self.id_idx_in_unified, :, :], prev_joints)
            # to use munkres package, we need int. munkres minimize cost, so use negative version
            scaled_distance_matrix = -nise_cfg.ALG.OKS_MULTIPLIER * dist_mat
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
    
    def est_joints(self, joint_detector):
        from nise_functions import _box2cs
        if not self.bboxes_unified and not self.is_first:
            # if first, no unified box because no flow to prop, so dont raise
            raise ValueError('Should unify bboxes first')
        self.joints = torch.zeros(self.unified_bboxes.shape[0], nise_cfg.DATA.num_joints, 2)  # FloatTensor
        joint_detector.eval()
        for i in range(self.unified_bboxes.shape[0]):
            bb = self.unified_bboxes[i, :4]  # no score
            cropped_human = imcrop(self.bgr_img, bb.long().numpy())  # np hwc, bgr
            cropped_human = np.array(cropped_human[:, :, ::-1])  # np hwc, rgb
            # debug_print('resizing ', self.img_name, '\'s', i)
            resized_human = resize(im_to_torch(cropped_human), *nise_cfg.DATA.human_bbox_size)  # tensor, chw, rgb
            resized_human.unsqueeze_(0)  # make it batch so we can use detector
            joint_hmap = joint_detector(resized_human)
            c, s = _box2cs(bb, self.img_ratio)  # TODO: is this right?
            s = np.ones([2])  # the scale above is like a 大傻逼
            # s = np.array([.7, 10])
            # make it batch so we can get right preds
            c_unsqueezed, s_unsqueezed = np.expand_dims(c, 0), np.expand_dims(s, 0)
            preds, _ = get_final_preds(
                simple_cfg, joint_hmap.detach().cpu().numpy(),
                c_unsqueezed, s_unsqueezed)  # bs x 16 x 2,  bs x 16 x 1
            # NG ! NG !
            self.joints[i, :, :] = to_torch(preds).squeeze()
        self.joints_detected = True
    
    def visualize(self, dataset, out_dir = 'images_out'):
        if self.id_assigned is False:
            raise ValueError('Should assign id first.')
        class_boxes = [[]] * 81
        # filter for showing joints
        class_boxes[1] = self.id_bboxes
        training_start_time = time.strftime("%H-%M-%S", time.localtime())
        
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
            thresh = nise_cfg.ALG.HUMAN_THRES,
            human_ids = self.human_ids,
            kp_thresh = 2,
            ext = 'jpg'
        )
        
        joints_to_show = self.joints[self.id_idx_in_unified]
        num_people, num_joints, _ = joints_to_show.shape
        
        for i in range(num_people):
            joints_i = joints_to_show[i, ...]  # 16 x 2
            joint_visible = torch.ones([num_joints, 1])
            nise_batch_joints = torch.cat([joints_i, joint_visible], 1)  # 16 x 3
            
            save_batch_image_with_joints(
                im_to_torch(self.bgr_img).unsqueeze(0),
                nise_batch_joints.unsqueeze(0),
                joint_visible.reshape([1, -1]),
                os.path.join(nise_cfg.DEBUG.JOINTS_DIR, self.img_name + "_joints_" + str(
                    self.human_ids[i].item()) + ".jpg"),
                nrow = 2
            )
            return
    
    def to_dict(self):
        '''
        Output string for json dump
        :return:
        '''
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
                    'annopoints': [
                        {
                            'x1': [0],
                            'x2': [0],
                            'y1': [0],
                            'y2': [0],
                            'point': [
                                {  # j for joints
                                    'id': [j],
                                    'x': [self.joints[i, j, 0].item()],
                                    'y': [self.joints[i, j, 1].item()],
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
