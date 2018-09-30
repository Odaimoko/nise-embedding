import nise_lib._init_paths
from tron_lib.core.test_for_pt import im_detect_all


class FrameItem:
    '''
        Item in Deque, storing image (for flow generation), joints(for propagation),
         trackids.
         One image or a batch of images?
    '''
    
    def __init__(self, is_first = False):
        '''

        :param is_first: is this frame the first of the sequence
        '''
        self.is_first = is_first
        self.img = None  # with original size? YES   ~~no, resize to some fixed size~~
        # num_person x num_joints x 2
        self.joints = None  # should be LongTensor since we are going to use them as index
        self.new_joints = None  # for similarity
        self.flow_to_current = None  # 2, h, w
        self.human_bboxes = None
        self.joint_prop_bboxes = None
        self.unified_bboxes = None
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
        cls_boxes = im_detect_all(detector, self.img)
        self.human_bboxes = cls_boxes[1]  # person is the first class of coco， 0 for background
        self.human_detected = True
    
    def gen_flow(self, prev_frame_img):
        # No precedent functions, root
        resized_img = resize(self.img, *nise_cfg.DATA.flow_input_size)
        resized_prev = resize(
            prev_frame_img, *nise_cfg.DATA.flow_input_size)
        ti = torch.stack([resized_prev, resized_img])
        self.flow_to_current = pred_flow(ti, flow_model)
        self.flow_calculated = True
    
    def joint_prop(self, prev_frame_joints):
        '''

        :param prev_frame_joints: 2 x num_people x num_joints
        :return:
        '''
        # preprocess
        if not ((prev_frame_joints.shape[2] == nise_cfg.DATA.num_joints and prev_frame_joints.shape[0] == 2) or (
                prev_frame_joints.shape[2] == 2 and prev_frame_joints.shape[1] == nise_cfg.DATA.num_joints)):
            raise ValueError(
                'Size not matched, current size ' + str(prev_frame_joints.shape))
        if prev_frame_joints.shape[2] == nise_cfg.DATA.num_joints:
            # :param prev_frame_joints: 2x num_people x num_joints
            prev_frame_joints = torch.transpose(prev_frame_joints, 0, 1)
            prev_frame_joints = torch.transpose(prev_frame_joints, 1, 2)
        
        if not self.flow_calculated:
            raise ValueError('Flow not calculated yet.')
        
        new_joints = torch.zeros(prev_frame_joints.shape)
        for person in range(new_joints.shape[0]):
            for joint in range(new_joints.shape[1]):
                joint_pos = prev_frame_joints[person, joint, :]  # x,y
                # x, y
                joint_flow = self.flow_to_current[:,
                             joint_pos[1], joint_pos[0]]
                new_joints[person, joint,
                :] = prev_frame_joints[person, joint, :].float() + joint_flow
        # for similarity
        self.new_joints = new_joints
        # calc new bboxes from new joints
        min_xs, _ = torch.min(new_joints[:, :, 0], 1)
        min_ys, _ = torch.min(new_joints[:, :, 1], 1)
        max_xs, _ = torch.max(new_joints[:, :, 0], 1)
        max_ys, _ = torch.max(new_joints[:, :, 1], 1)
        # extend
        ws = max_xs - min_xs
        hs = max_ys - min_ys
        ws = ws * nise_cfg.DATA.bbox_extend_factor[0]
        hs = hs * nise_cfg.DATA.bbox_extend_factor[1]
        min_xs -= ws
        max_xs += ws
        min_ys -= hs
        max_xs += hs
        min_xs.clamp_(0, nise_cfg.DATA.flow_input_size[0])
        max_xs.clamp_(0, nise_cfg.DATA.flow_input_size[0])
        min_ys.clamp_(0, nise_cfg.DATA.flow_input_size[1])
        max_ys.clamp_(0, nise_cfg.DATA.flow_input_size[1])
        
        self.joint_prop_bboxes = torch.stack([
            min_xs, min_ys, max_xs, max_ys
        ], 1)
        # assert(self.joint_prop_bboxes.shape )
        self.joints_proped = True
    
    def unify_bbox(self):
        if not self.joints_proped or not self.human_detected:
            raise ValueError(
                'Should run human detection and joints propagation first')
        all_bboxes = torch.stack(self.human_bboxes, self.joint_prop_bboxes)
        self.bboxes_unified = True
        raise NotImplementedError
    
    def assign_id(self, Q, dist_func = None):
        
        # ─── ASSOCIATE IDS USING GREEDY MATCHING ────────────────────────────────────────
        # ────────────────────────────────────────────────────────────────────────────────
        """ input: distance matrix; output: correspondence   """
        if not self.joints_detected:
            raise ValueError('Should detect joints first')
        self.human_ids = []
        if self.is_first:
            # if it's the first frame, just assign every
            for i in range(self.human_bboxes.shape[0]):
                self.human_ids.append(i)
        else:
            if dist_func is None: raise NotImplementedError('Should pass a distance function in')
            prev_frame_joints = Q[-1].joints
            prev_ids = Q[-1].human_ids
            num_human_prev = prev_frame_joints.shape[0]
            num_human_cur = self.joints.shape[0]
            distance_matrix = torch.zeros(num_human_prev, num_human_cur)
            for prev in range(num_human_prev):
                for cur in range(num_human_cur):
                    distance_matrix[prev, cur] = dist_func(prev_frame_joints[prev], self.joints[cur])
            raise NotImplementedError
        self.id_assigned = True
    
    def est_joints(self, joint_detector):
        if not self.bboxes_unified:
            raise ValueError('Should unify bboxes first')
        
        self.joints_detected = True
        
        raise NotImplementedError
