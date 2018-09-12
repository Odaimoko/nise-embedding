from __future__ import print_function, absolute_import

import time

import torch
import torch.nn.parallel
import torch.optim
import visdom
from torchnet import meter
# import resnet
# from pose import Bar
# from pose.utils.logger import Logger, savefig
# from pose.utils.evaluation import accuracy, AverageMeter, final_preds
# from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
# from pose.utils.osutils import mkdir_p, isfile, isdir, join
# from pose.utils.imutils import batch_with_heatmap
# from pose.utils.transforms import fliplr, flip_back
# import pose.models as models
# import pose.datasets as datasets
from tqdm import tqdm

from ae_lib.models.posenet import PoseNet
from datasetpy.mpiivideo import MPIIDataset
from datasetpy.mscocoMulti import MscocoMulti
from mpii_config import cfg, get_arg_parser
from coco_config import cfg as coco_cfg
from utils.misc import save_checkpoint
from utils.visualize import Visualizer


def tag_loss_batch(embedding, track_id, joints, sigma = 1):
    '''
        Compute pull and push loss in a batch
        For now regard treat samples indepentdently
    :param embedding: bs x len_embed x *heatmap_size
    :param track_id: bs
    :param gt_joints_pos: bs x num_joints x 3, within heatmap size
    :param sigma: I dont know what for, so set to 1
    :return:
    '''
    # TODO: what if within a batch a track_id appears more than once
    batch_size = embedding.size(0)
    gt_joints_pos = joints[:, :, :2]
    gt_joints_visible = joints[:, :, 2] > 0
    num_joints = joints.size(0)
    loss_pull = 0.
    loss_push = 0.
    ref_embs = []
    for person in range(batch_size):
        single_emb, t_id, gt_pos = embedding[person], track_id[person], gt_joints_pos[person]
        # corresponding gt poses' embedding
        gt_pos_emb = single_emb[:, gt_pos[:, 0], gt_pos[:, 1]]  # len_embed x num_joints
        # emb for a single person
        reference_embedding = torch.mean(gt_pos_emb, dim = 1)  # len_embed
        ref_embs.append(reference_embedding)  # used to calc push loss
        # calc loss... # len_embed  x num_joints,unsqueeze for broadcasting
        diff_ref_joint = gt_pos_emb - reference_embedding.unsqueeze(1)
        squared_diff = diff_ref_joint * diff_ref_joint
        # TODO: filter out invisible joints, or should we?
        single_tag_loss = torch.sum(squared_diff, dim = 0) * gt_joints_visible.float()
        loss_pull += torch.sum(single_tag_loss)
    loss_pull = loss_pull / num_joints / batch_size  # normalize
    
    ref_embs = torch.stack(ref_embs)  # bs x len_embed
    for i in range(batch_size):
        diff_between_people = ref_embs - ref_embs[i, :]  # len_embed  x num_joints
        squared_diff = diff_between_people * diff_between_people  # len_embed  x num_joints
        e = torch.exp(- torch.sum(squared_diff, dim = 1) / (2 * sigma ** 2))  # num_joints
        loss_push += torch.sum(e)  # scalar
    loss_push -= batch_size  # if regard the batch as diff people, the same people shouldnt be calced in push loss
    loss_push = loss_push / batch_size / batch_size
    return loss_pull, loss_push


def plot_all_loss_with_vis_set(vis_set, update):
    if vis_set is None: return
    viz, loss_meters = vis_set[0], vis_set[1]
    for k, v in loss_meters.items():
        v.add(update[k])  # 为了可视化增加的内容
    viz.plot_many_stack({k: v.value()[0] for k, v in loss_meters.items()})


def train(epoch, data_loader, model, optimizer, vis_set = None):
    model.train()
    i = -1
    for person in tqdm(data_loader):
        i += 1  # Cant use enumerate, so = =+
        # if i > 3: break
        data, gt_heatmaps, head_bbox, joints_px, track_id = \
            person['data'], person['gt_heatmaps'], person['head_bbox'], person['joints'], person['track_id']
        # Get grad of input
        
        data.require_grad = True
        if torch.cuda.is_available():
            data = data.cuda()
            gt_heatmaps = gt_heatmaps.cuda()
            head_bbox = head_bbox.cuda()
            joints_px = joints_px.cuda()
            track_id = track_id.cuda()
        
        batch_size = data.size(0)
        
        result = model(data)  # torch.Size([bs, num_stack_hg, oup_dim, 64, 64])
        joints_for_hmap = joints_px / data_loader.dataset.feat_stride
        joints_for_hmap[:, :, 2] = joints_px[:, :, 2]
        hourglass_final_layer_id = net_cfg['nstack'] - 1
        num_joints = net_cfg['num_parts']
        pred_heatmaps = result[:, hourglass_final_layer_id, 0:num_joints]  # bs x num_joints x *heatmap_size
        # 这里是分开来的， 训练的tag并没有帮助到heatmap
        tags = result[:, hourglass_final_layer_id, num_joints:]  # bs x len_emb x *heatmap_size
        
        optimizer.zero_grad()
        
        pull_tag_loss, push_tag_loss = tag_loss_batch(tags, track_id, joints_for_hmap)
        
        # heatmap_losses = []  #
        heatmap_losses = 0.  # add them all
        for b in range(batch_size):
            # gt_heatmaps = torch.Size([bs, num_sigmas, num_joints, 64, 64])
            # Average over all sigmas
            gt_heatmap = gt_heatmaps[b].mean(0)
            pred_heatmap = pred_heatmaps[b]
            if torch.cuda.is_available():
                gt_heatmap = gt_heatmap.cuda()
            heatmap_losses += heatmap_loss_func(pred_heatmap, gt_heatmap)
            # for sig in gt_heatmap:  # sig: num_joints x * hmap_size
            #     heatmap_losses += heatmap_loss_func(pred_heatmap, sig)
            # for gt_heatmap in gt_heatmaps:
            #     gt_heatmap = gt_heatmap.cuda()
            #     heatmap_losses.append(heatmap_loss_func(heatmap_joints, gt_heatmap))
            #
        # normalize, loss = loss / (batch size) / (num joints) or should we?
        heatmap_losses = heatmap_losses / (gt_heatmaps.shape[0] * gt_heatmaps.shape[2])
        
        # I don't know the meaning of weight tag losses, since they are treated differently?
        # Oh no, they are both related to hourglass, alright ...
        loss = heatmap_losses + cfg.emb_tag_weight * (pull_tag_loss + push_tag_loss)
        
        plot_all_loss_with_vis_set(vis_set, {
            'total loss': loss.item(),
            'pull loss': pull_tag_loss.item(),
            'push loss': push_tag_loss.item(),
            'hmap loss': heatmap_losses.item(),
        })
        loss.backward()
        optimizer.step()
        print('\n')
    
    if vis_set:
        viz = vis_set[2]
        data.require_grad = False
        gt_heatmaps.require_grad = False
        viz_plot_gt_and_pred(viz, data, gt_heatmaps[:, 0, ...], pred_heatmaps)


# set nstack to 4 will
net_cfg = {'nstack': 2, 'inp_dim': 256, 'oup_dim': 32,
           'num_parts': 16, 'increase': 128, 'keys': ['imgs']}

net = PoseNet(**net_cfg).cuda()
args = get_arg_parser().parse_args()


# train_loader = torch.utils.data.DataLoader(
#     MscocoMulti(coco_cfg),
#     batch_size = cfg.batch_size * args.num_gpus, shuffle = True,
#     num_workers = args.workers, pin_memory = True)

mpii_train_loader = torch.utils.data.DataLoader(
    MPIIDataset( 'train', cfg),
    batch_size = cfg.batch_size * args.num_gpus, shuffle = True,
    num_workers = args.workers, pin_memory = True)

training_start_time = time.strftime("%m-%d_%H:%M", time.localtime())
optimizer = torch.optim.SGD(net.parameters(), lr = cfg.lr, weight_decay = cfg.weight_decay)
# AE's heatmap loss considers COCO's mask to rule out crowd, here we dont
heatmap_loss_func = torch.nn.MSELoss().cuda()

vizer = Visualizer(env = 'AE_loss')
viz = visdom.Visdom(env = 'AE_heatmap')

loss_meters = {
    'total loss': meter.AverageValueMeter(),
    'pull loss': meter.AverageValueMeter(),
    'push loss': meter.AverageValueMeter(),
    'hmap loss': meter.AverageValueMeter(),
}
vis_set = (vizer, loss_meters, viz)

for p in net.parameters():
    p.require_grad = True
from utils.visualize import viz_plot_gt_and_pred

for epoch in range(cfg.START_EPOCH, cfg.END_EPOCH):
    train(epoch, mpii_train_loader, net, optimizer, vis_set)
    
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, torch.Tensor(0), False, checkpoint = "checkpoint",
        filename = "gnet-" + training_start_time + "-epoch-" + str(epoch + 1) + ".pth")


# for i, (inputs, targets, valid, meta) in enumerate(train_loader):
#     '''
#       这个loader的特殊之处在于，进来的每一个inputs都是一个人（obj），所以所有关节的id已经统一了。
#       但是posetrack就不一样
#     '''
#     if i > 3:
#         break
#     to_net = inputs.permute(0, 2, 3, 1).cuda()
#     # targets = targets.cuda()
#     print(to_net.shape)
#     result = net(to_net)  # torch.Size([bs, nStack, 68, 64, 48])
#     # labels = {
#     #   'keypoints': torch.rand([1, 30, 17, 2]),
#     #   'heatmaps': torch.rand([1, 17, 128, 128]),
#     #   'masks': torch.rand([1, 128, 128])
#     # }
#     # push_loss, pull_loss, joint_loss = net.calc_loss(result,**labels )
#
#     heatmap_losses = []
#     hourglass_final_layer_id = net_cfg['nstack'] - 1
#     num_joints = net_cfg['num_parts']
#     heatmap_joints = result[:, hourglass_final_layer_id, 0:num_joints]
#     tags = result[:, hourglass_final_layer_id, num_joints:]
#
#     for gt_heatmap in targets:
#         gt_heatmap = gt_heatmap.cuda()
#         heatmap_losses.append(heatmap_loss_func(heatmap_joints, gt_heatmap))
#
#     print(i, inputs.shape, valid.shape)
