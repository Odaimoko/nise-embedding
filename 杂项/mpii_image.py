
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
from datasetpy.mpii import Mpii
from mpii_config import cfg, get_arg_parser
from coco_config import cfg as coco_cfg
from nise_utils.misc import save_checkpoint
from nise_utils.visualize import Visualizer


def train_mpii_video(epoch, data_loader, model, optimizer, vis_set = None):
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


net_cfg = {'nstack': 2, 'inp_dim': 256, 'oup_dim': 32,
           'num_parts': 16, 'increase': 128, 'keys': ['imgs']}

net = PoseNet(**net_cfg).cuda()
args = get_arg_parser().parse_args()


mpii_train_loader = torch.utils.data.DataLoader(
    Mpii('data/mpii-image/mpii_annotations.json','data/mpii-image/images',cfg.data_shape[0],cfg.output_shape[0]),
    batch_size = cfg.batch_size * args.num_gpus, shuffle = True,
    num_workers = args.workers, pin_memory = True)

for p in net.parameters():
    p.require_grad = True
for i, (inputs, target, meta) in enumerate(mpii_train_loader):
    print(person)