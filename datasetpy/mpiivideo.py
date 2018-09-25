from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from nise_utils.visualize import viz_plot_gt_and_pred, plt_2_viz


def gaussian_ground_truth(feat_stride, heatmap_size, joints_np, sigmas = (7)):
    # DONE: even if it's not visible, still give it a hmap, for now
    '''
    
    :param feat_stride:
    :param heatmap_size:
    :param joints_np:
    :param sigmas:
    :return: num_sigmas x num_joints x * hmap_size
    '''
    joints_np = joints_np.copy()
    # cast joints to be in heatmap_size
    joints_np[:, :2] /= feat_stride
    
    def single_gaussian_gt(pt):
        hmaps = []
        for sigma in sigmas:
            # generate heatmaps for different kernel size
            hmap = np.zeros(heatmap_size)
            # In pt, y comse first
            if pt[2]:
                # if visible
                hmap[int(pt[0])][int(pt[1])] = 1
                hmap = cv2.GaussianBlur(hmap, (sigma, sigma), 0)
                am = np.amax(hmap)
                hmap /= am / 255
            hmaps.append(hmap)
        hmaps_np = np.stack(hmaps)
        return hmaps_np
    
    hmaps = [single_gaussian_gt(pos) for pos in joints_np]
    hmaps = np.stack(hmaps)
    hmaps = np.swapaxes(hmaps, 0, 1)
    return hmaps


def flip_coin():
    return random.randint(0, 1)


class MPIIDataset(Dataset):
    MODE = ['train', 'test', 'val']
    # 'image': once a image, with all annotations heatmaps gatherd to be a single heatmap
    # 'person' once a person, with the original image and heatmaps of only this person
    GET_ITEM_MODE = ['image', 'person']
    
    def __init__(self, mode, cfg):
        super(MPIIDataset, self).__init__()
        
        if mode not in MPIIDataset.MODE:
            raise Exception('Mode must be in ' + str(MPIIDataset.MODE))
        self.mode = mode
        self.data_shape = cfg.data_shape
        self.output_shape = cfg.output_shape
        self.feat_stride = self.data_shape[0] / self.output_shape[0]
        self.data_path = cfg.image_path
        self.person_anno, self.frame_with_no_anno, self.person_only_head = self.process_anno()
        self.flip_images = False
        self.rotate_images = False
        self.transform = None
        # use different sigmas to adapt to different scale of people
        self.gt_hmap_sigmas = cfg.gt_hmap_sigmas  # sigma when generating 2d gaussian heatmaps,
        self.num_joints = cfg.num_class  # actually joints id 6 and 7 are not annotated in this dataset, so only 14 avaliable
        self.max_theta = 45
        self.get_item_mode = 'person'
    
    def process_anno(self):
        with open(os.path.join(self.data_path, 'mpii-video-annotations.json'), 'r') as f:
            anno = json.load(f)
        MAX_PEOPLE = 100
        no_anno = []
        anno_list = []
        only_head = []
        for i, video in enumerate(anno):
            video = video[0]
            for j, frame in enumerate(video):
                # for the sake of unique track_id among all sequences
                if 'annorect' not in frame.keys() or frame['annorect'] is None:
                    # some image are not annotated
                    no_anno.append(frame)
                    continue
                
                for k, person in enumerate(frame['annorect']):
                    # record image file in person
                    img = frame['image']
                    if 'annopoints' not in person.keys() or person['annopoints'] is None:
                        # some people dont have annatations, their track_id's are all -1
                        only_head.append(person)
                        continue
                    person['annopoints'] = person['annopoints']['point']
                    try:
                        del person['score']
                        del person['scale']
                    except KeyError:
                        pass
                    person['track_id'] = person['track_id'] + i * MAX_PEOPLE
                    person.update({"image": img['name']})
                    anno_list.append(person)
        return anno_list, no_anno, only_head
    
    def split_db(self):
        """
            split all annotations to train/val set
            Or should we?
        """
    
    def _augment_image(self, img, joints):
        h, w, channels = img.shape
        center = (h / 2, w / 2)
        
        if self.flip_images and flip_coin():
            # I dont know if we should flip all sequence, I think so, so skip this for now
            pass
        if self.rotate_images and flip_coin():
            # A little bit ...
            # theta = random.uniform(0, self.max_theta)
            theta = 10
            rot_mat = cv2.getRotationMatrix2D(center, theta, 1.0)
            img_rotated = cv2.warpAffine(img, rot_mat, (h, w))  # montainai
            
            # deal with joints
            joints_rotated = np.zeros_like(joints)
            for i in range(self.num_joints):
                y, x, visible = joints[i]
                coordinate = np.array([x, y, 1])  # Homogeneous coordinates
                if not visible:
                    continue
                if x >= 0 and y >= 0:
                    coordinate_rotated = rot_mat.dot(coordinate.T)  # should be vector of size 2
                    new_x, new_y = coordinate_rotated
                    visible_rotated = visible * (new_x >= 0 and new_y >= 0 and new_x <= w and new_y <= h)
                    # Here we should exchange y and x ...
                    joints_rotated[i, :2] = [coordinate_rotated[1], coordinate_rotated[0]]
                    joints_rotated[i, 2] = visible_rotated
            img = img_rotated
            joints = joints_rotated
        
        return img, joints
    
    def _resize_image(self, img_np, joints_np):
        
        # ALright always think this is to violent brutal...
        resized_img = cv2.resize(img_np, self.data_shape)
        
        # clamp joints to be in 0~img_size
        h, w, channel = img_np.shape
        h_ratio = h / self.data_shape[0]
        w_ratio = w / self.data_shape[1]
        joints_np[:, 0] /= h_ratio
        joints_np[:, 0] = np.floor(joints_np[:, 0])
        joints_np[:, 1] /= w_ratio
        joints_np[:, 1] = np.floor(joints_np[:, 1])
        
        return resized_img, joints_np
    
    def __len__(self):
        # for m in MPIIDataset.MODE:
        #     if self.mode == m:
        #         return 0:
        # return super().__len__()
        return len(self.person_anno)
        # return 0
    
    def __getitem__(self, index):
        person = self.person_anno[index]
        image_file_path = os.path.join(self.data_path, person['image'])
        img_np = cv2.imread(image_file_path, cv2.IMREAD_COLOR |
                            cv2.IMREAD_IGNORE_ORIENTATION)  # ndarray, h x w x 3
        if img_np is None:
            raise ValueError('Failed to read {}'.format(image_file_path))
        # Since openCV use BGR, need to inverse it
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        # cannot use the following, will raise
        # ERROR some of the strides of a given numpy array are negative.
        # img_np = img_np[:, :, ::-1]
        
        joints_anno = person['annopoints']
        joints_np = np.zeros([self.num_joints, 3])
        for j in joints_anno:
            # Note here we use y first, since height comes first
            joints_np[j['id'], :] = np.array([j['y'], j['x'], j['is_visible']])
        
        # simply resize to img_size, scale joints_np linearly
        resized_img, joints_np = self._resize_image(img_np, joints_np)
        
        if self.mode == 'train':
            # Only augment data when training.
            # TODO: can do some data augmentation, eg. scale, rotation[tick], flip
            # when applying those, joints must be processed too
            # I dont think tracking needs flipping, for now
            resized_img, joints_np = self._augment_image(resized_img, joints_np)
        
        # after conv, the downsample rate is feat_stride
        gt_heatmaps = gaussian_ground_truth(self.feat_stride, self.output_shape, joints_np, self.gt_hmap_sigmas)
        
        # All to float32 to avoid type confliction
        # self.num_joints x heatmapsize
        target = torch.from_numpy(gt_heatmaps).float()
        # convert to C x H x W
        img = torch.from_numpy(plt_2_viz(resized_img)).float()
        head_bbox = torch.tensor([person['x1'], person['x2'], person['y1'], person['y2']]).float()
        joints = torch.from_numpy(joints_np).long()
        person.update({
            'gt_heatmaps': target,
            'data': img,
            'original': img_np,
            'head_bbox': head_bbox,
            'joints': joints,
        })
        del person['x1']
        del person['x2']
        del person['y1']
        del person['y2']
        if not __name__ == '__main__':
            del person['original']  # must be same size so save it only when debuging
            del person['image']  # str infomation eliminate
            del person['annopoints']  # only need gt_heatmaps to train
        # return person['data'], person['gt_heatmaps'],person['track_id'],person['']
        return person


if __name__ == '__main__':
    NUM_IMG_SHOWN_DEBUG = 1
    
    import visdom
    
    viz = visdom.Visdom()
    
    d = MPIIDataset('../data/mpii-video-pose', 'train')
    d.rotate_images = True
    d1 = d[1113]
    to_viz = plt_2_viz(d1['data'])
    # viz.image(plt_2_viz(d1['original']))
    viz.image(to_viz)
    # to_viz /= torch.max(to_viz)
    # gt_heatmaps = np.zeros ((d.num_joints,d.img_size[0],d.img_size[1]))
    # for i in range(gt_heatmaps.shape[0]):
    #
    #     gt_heatmaps[i,...] = cv2.resize(d1['gt_heatmaps'][i,...].numpy(),d.img_size)
    # viz_plot_gt_and_pred(viz, to_viz.unsqueeze(0),np.expand_dims(gt_heatmaps ,0), None)
    viz_plot_gt_and_pred(viz, to_viz.unsqueeze(0), np.expand_dims(d1['gt_heatmaps'].numpy(), 0), None)
    
    print('TEST DONE')
