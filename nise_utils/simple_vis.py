# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2

from nise_lib.nise_config import nise_cfg


# from core.inference import get_max_preds

def save_single_whole_image_with_joints(torch_img, src_joints,
                                        file_name, nrow = 1, padding = 1, boxes = None,
                                        thresh = nise_cfg.DEBUG.VIS_HUMAN_THRES, human_ids = None):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(torch_img, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()
    if boxes is not None:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)
        
        if human_ids is not None:
            assert (boxes.shape[0] == human_ids.shape[0])
        
        for i in sorted_inds:
            bbox = boxes[i, :4]
            score = boxes[i, -1]
            human_id = human_ids[i].item() if human_ids is not None else ''
            if score < thresh:
                continue
            
            # print(dataset.classes[classes[i]], score, bbox, human_id.item())
            cv2.rectangle(ndarr, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), (0, 0, 255), thickness = 2)
            
            cv2.putText(ndarr, 'ID: ' + str(human_id), (int(bbox[0]), int(bbox[1]) - 2),
                        cv2.FONT_HERSHEY_COMPLEX,
                        .6, (255, 255, 255), thickness = 2)
    
    num_people, num_joints, _ = src_joints.shape
    colores = [
        # red + black
        (112, 18, 52,),
        (247, 29, 53,),
        (45, 52, 62,),
        # blue + green
        (13, 114, 113),
        (11, 110, 79),
        (8, 160, 69),
        (107, 191, 89),
        (221, 183, 113),
        # purple
        (84, 74, 155),
        (162, 163, 187),
        (147, 149, 211),
        (179, 183, 238),
        (251, 249, 255),
        # pink + yellow
        (199, 234, 228),
        (167, 232, 189),
        (252, 188, 184),
        (239, 167, 167),
        (255, 217, 114)
    ]
    
    joint_id_offset = [(5, 5), (-5, -5)]
    for k in range(num_people):
        joints = src_joints[k]
        joints_vis = joints[:, 2]
        if human_ids is not None:
            color_k = human_ids[k].item() if human_ids is not None else k
            color_k = color_k % len(colores)
        else:
            color_k = k
        # print(k, color_k, human_ids[k])
        for i, (joint, joint_vis) in enumerate(zip(joints, joints_vis)):
            if joint_vis.item():
                # use id as index to get color, if no id is presented, use k.
                
                cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, colores[color_k], 2)
                cv2.putText(ndarr, str(i), (int(joint[0]) + joint_id_offset[k % len(joint_id_offset)][0],
                                            int(joint[1]) + joint_id_offset[k % len(joint_id_offset)][1]),
                            cv2.FONT_HERSHEY_COMPLEX,
                            .6, (0, 0, 255), 1)
    cv2.imwrite(file_name, ndarr)


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow = 8, padding = 2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()
    
    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]
            
            for i, (joint, joint_vis) in enumerate(zip(joints, joints_vis)):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis.item():
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 4, [255, 0, 0], 2)
                    cv2.putText(ndarr, str(i), (int(joint[0]), int(joint[1])), cv2.FONT_HERSHEY_COMPLEX,
                                1, (0, 0, 255), 1)
            
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def get_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis, nrow = 8, padding = 2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()
    
    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]
            
            for i, (joint, joint_vis) in enumerate(zip(joints, joints_vis)):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis.item():
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 4, [255, 0, 0], 2)
                    cv2.putText(ndarr, str(i), (int(joint[0]), int(joint[1])), cv2.FONT_HERSHEY_COMPLEX,
                                1, (0, 0, 255), 1)
            
            k = k + 1
    return ndarr

# def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
#                         normalize=True):
#     '''
#     batch_image: [batch_size, channel, height, width]
#     batch_heatmaps: ['batch_size, num_joints, height, width]
#     file_name: saved file name
#     '''
#     if normalize:
#         batch_image = batch_image.clone()
#         min = float(batch_image.min())
#         max = float(batch_image.max())
#
#         batch_image.add_(-min).div_(max - min + 1e-5)
#
#     batch_size = batch_heatmaps.size(0)
#     num_joints = batch_heatmaps.size(1)
#     heatmap_height = batch_heatmaps.size(2)
#     heatmap_width = batch_heatmaps.size(3)
#
#     grid_image = np.zeros((batch_size*heatmap_height,
#                            (num_joints+1)*heatmap_width,
#                            3),
#                           dtype=np.uint8)
#
#     preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())
#
#     for i in range(batch_size):
#         image = batch_image[i].mul(255)\
#                               .clamp(0, 255)\
#                               .byte()\
#                               .permute(1, 2, 0)\
#                               .cpu().numpy()
#         heatmaps = batch_heatmaps[i].mul(255)\
#                                     .clamp(0, 255)\
#                                     .byte()\
#                                     .cpu().numpy()
#
#         resized_image = cv2.resize(image,
#                                    (int(heatmap_width), int(heatmap_height)))
#
#         height_begin = heatmap_height * i
#         height_end = heatmap_height * (i + 1)
#         for j in range(num_joints):
#             cv2.circle(resized_image,
#                        (int(preds[i][j][0]), int(preds[i][j][1])),
#                        1, [0, 0, 255], 1)
#             heatmap = heatmaps[j, :, :]
#             colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#             masked_image = colored_heatmap*0.7 + resized_image*0.3
#             cv2.circle(masked_image,
#                        (int(preds[i][j][0]), int(preds[i][j][1])),
#                        1, [0, 0, 255], 1)
#
#             width_begin = heatmap_width * (j+1)
#             width_end = heatmap_width * (j+2)
#             grid_image[height_begin:height_end, width_begin:width_end, :] = \
#                 masked_image
#             # grid_image[height_begin:height_end, width_begin:width_end, :] = \
#             #     colored_heatmap*0.7 + resized_image*0.3
#
#         grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image
#
#     cv2.imwrite(file_name, grid_image)
#
#
# def save_debug_images(config, input, meta, target, joints_pred, output,
#                       prefix):
#     if not config.DEBUG.DEBUG:
#         return
#
#     if config.DEBUG.SAVE_BATCH_IMAGES_GT:
#         save_batch_image_with_joints(
#             input, meta['joints'], meta['joints_vis'],
#             '{}_gt.jpg'.format(prefix)
#         )
#     if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
#         save_batch_image_with_joints(
#             input, joints_pred, meta['joints_vis'],
#             '{}_pred.jpg'.format(prefix)
#         )
#     if config.DEBUG.SAVE_HEATMAPS_GT:
#         save_batch_heatmaps(
#             input, target, '{}_hm_gt.jpg'.format(prefix)
#         )
#     if config.DEBUG.SAVE_HEATMAPS_PRED:
#         save_batch_heatmaps(
#             input, output, '{}_hm_pred.jpg'.format(prefix)
#         )
