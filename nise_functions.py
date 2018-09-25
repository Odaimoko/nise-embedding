import flow_models
import flow_losses
import flow_datasets
import torch
from torch import nn
from nise_debugging_func import gen_rand_flow
import os
from tensorboardX import SummaryWriter
import colorama
import argparse


def flow_init_parser_and_tools(parser, tools):
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--total_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', '-b', type=int,
                        default=8, help="Batch size")
    parser.add_argument('--train_n_batches', type=int, default=-1,
                        help='Number of min-batches per epoch. If < 0, it will be determined by training_dataloader')
    parser.add_argument('--crop_size', type=int, nargs='+', default=[256, 256],
                        help="Spatial dimension to crop training samples for training")
    parser.add_argument('--gradient_clip', type=float, default=None)
    parser.add_argument('--schedule_lr_frequency', type=int, default=0,
                        help='in number of iterations (0 for no schedule)')
    parser.add_argument('--schedule_lr_fraction', type=float, default=10)
    parser.add_argument("--rgb_max", type=float, default=255.)

    parser.add_argument('--number_workers', '-nw',
                        '--num_workers', type=int, default=8)
    parser.add_argument('--number_gpus', '-ng', type=int,
                        default=-1, help='number of GPUs to use')
    parser.add_argument('--no_cuda', action='store_true')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--name', default='run', type=str,
                        help='a name to append to the save directory')
    parser.add_argument('--save', '-s', default='./work',
                        type=str, help='directory for saving')

    parser.add_argument('--validation_frequency', type=int,
                        default=5, help='validate every n epochs')
    parser.add_argument('--validation_n_batches', type=int, default=-1)
    parser.add_argument('--render_validation', action='store_true',
                        help='run inference (save flows to file) and every validation_frequency epoch')

    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--inference_size', type=int, nargs='+', default=[-1, -1],
                        help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
    parser.add_argument('--inference_batch_size', type=int, default=1)
    parser.add_argument('--inference_n_batches', type=int, default=-1)
    parser.add_argument('--save_flow', action='store_true',
                        help='save predicted flows to file')

    parser.add_argument('--flownet_resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--log_frequency', '--summ_iter',
                        type=int, default=1, help="Log every n batches")

    parser.add_argument('--fp16', action='store_true',
                        help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--fp16_scale', type=float, default=1024.,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    # ─── TOOLS ──────────────────────────────────────────────────────────────────────

    tools.add_arguments_for_module(
        parser, flow_models, argument_for_class='model', default='FlowNet2')

    tools.add_arguments_for_module(
        parser, flow_losses, argument_for_class='loss', default='L1Loss')

    tools.add_arguments_for_module(parser, torch.optim, argument_for_class='optimizer', default='Adam',
                                   skip_params=['params'])

    tools.add_arguments_for_module(parser, flow_datasets, argument_for_class='training_dataset',
                                   default='MpiSintelFinal',
                                   skip_params=['is_cropped'],
                                   parameter_defaults={'root': './MPI-Sintel/flow/training'})

    tools.add_arguments_for_module(parser, flow_datasets, argument_for_class='validation_dataset',
                                   default='MpiSintelClean',
                                   skip_params=['is_cropped'],
                                   parameter_defaults={'root': './MPI-Sintel/flow/training',
                                                       'replicates': 1})

    tools.add_arguments_for_module(parser, flow_datasets, argument_for_class='inference_dataset',
                                   default='MpiSintelClean',
                                   skip_params=['is_cropped'],
                                   parameter_defaults={'root': './MPI-Sintel/flow/training',
                                                       'replicates': 1})


def load_flow_model(args, parser, tools):
    # Parse the official arguments
    with tools.TimerBlock("Parsing Arguments") as block:
        if args.number_gpus < 0:
            args.number_gpus = torch.cuda.device_count()

        # Get argument defaults (hastag #thisisahack)
        parser.add_argument('--IGNORE', action='store_true')
        defaults = vars(parser.parse_args(['--IGNORE']))

        # Print all arguments, color the non-defaults
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
            block.log('{}{}: {}{}'.format(color, argument, value, reset))

        args.model_class = tools.module_to_dict(flow_models)[args.model]
        args.optimizer_class = tools.module_to_dict(torch.optim)[
            args.optimizer]
        args.loss_class = tools.module_to_dict(flow_losses)[args.loss]

        args.training_dataset_class = tools.module_to_dict(flow_datasets)[
            args.training_dataset]
        args.validation_dataset_class = tools.module_to_dict(flow_datasets)[
            args.validation_dataset]
        args.inference_dataset_class = tools.module_to_dict(flow_datasets)[
            args.inference_dataset]

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        args.log_file = os.path.join(args.save, 'args.txt')

        # dict to collect activation gradients (for training debug purpose)
        args.grads = {}

        if args.inference:
            args.skip_validation = True
            args.skip_training = True
            args.total_epochs = 1
            args.inference_dir = "{}/inference".format(args.save)

    # Dynamically load model and loss class with parameters passed in
    # via "--model_[param]=[value]" or "--loss_[param]=[value]" arguments
    with tools.TimerBlock("Building {} model".format(args.model)) as block:
        class ModelAndLoss(nn.Module):
            def __init__(self, args):
                super(ModelAndLoss, self).__init__()
                kwargs = tools.kwargs_from_args(args, 'model')
                self.model = args.model_class(args, **kwargs)
                kwargs = tools.kwargs_from_args(args, 'loss')
                self.loss = args.loss_class(args, **kwargs)

            def forward(self, data):
                output = self.model(data)
                return output
                # loss_values = self.loss(output, target)
                #
                # if not inference:
                #     return loss_values
                # else:
                #     return loss_values, output

        model_and_loss = ModelAndLoss(args)

        block.log('Number of parameters: {}'.format(
            sum([p.data.nelement() if p.requires_grad else 0 for p in model_and_loss.parameters()])))

        # assing to cuda or wrap with dataparallel, model and loss
        if args.cuda and (args.number_gpus > 0) and args.fp16:
            block.log('Parallelizing')
            model_and_loss = nn.parallel.DataParallel(
                model_and_loss, device_ids=list(range(args.number_gpus)))

            block.log('Initializing CUDA')
            model_and_loss = model_and_loss.cuda().half()
            torch.cuda.manual_seed(args.seed)
            param_copy = [param.clone().type(torch.cuda.FloatTensor).detach()
                          for param in model_and_loss.parameters()]

        elif args.cuda and args.number_gpus > 0:
            block.log('Initializing CUDA')
            model_and_loss = model_and_loss.cuda()
            block.log('Parallelizing')
            model_and_loss = nn.parallel.DataParallel(
                model_and_loss, device_ids=list(range(args.number_gpus)))
            torch.cuda.manual_seed(args.seed)

        else:
            block.log('CUDA not being used')
            torch.manual_seed(args.seed)

        # Load weights if needed, otherwise randomly initialize
        # 要用这个
        if args.flownet_resume and os.path.isfile(args.flownet_resume):
            block.log("Loading checkpoint '{}'".format(args.flownet_resume))
            checkpoint = torch.load(args.flownet_resume)
            if not args.inference:
                args.start_epoch = checkpoint['epoch']
            best_err = checkpoint['best_EPE']
            model_and_loss.module.model.load_state_dict(
                checkpoint['state_dict'])
            block.log("Loaded checkpoint '{}' (at epoch {})".format(
                args.flownet_resume, checkpoint['epoch']))

        block.log("Initializing save directory: {}".format(args.save))
        if not os.path.exists(args.save):
            os.makedirs(args.save)

        # train_logger = SummaryWriter(log_dir = os.path.join(
        #     args.save, 'train'), comment = 'training')
        # validation_logger = SummaryWriter(log_dir = os.path.join(
        #     args.save, 'validation'), comment = 'validation')

    return model_and_loss


# Reusable function for inference
def pred_flow(two_images, model):
    '''
        Already cudaed
    :param two_images: channels, 2, h, w
    :param model:
    :return:
    '''
    model.eval()

    c, _, h, w = two_images.shape
    with torch.no_grad():
        # data[0] torch.Size([8, 3, 2, 384, 1024]) ，bs x channels x num_images, h, w
        # target torch.Size([8, 2, 384, 1024]) maybe offsets
        # losses: list (2)
        # output: torch.Size([bs, 2, 384, 1024])
        two_images = torch.unsqueeze(two_images, 0)  # batchize
        # losses, output = model(
        #     data=two_images, target=gen_rand_flow(1, h, w), inference=True)
        output = model(two_images)

    return output


def load_simple_model():
    from pose_estimation import _init_paths
    from core.config import config as simple_cfg
    from core.config import update_config
    from core.config import update_dir
    from core.config import get_model_name
    from core.inference import get_final_preds
    from core.loss import JointsMSELoss
    from nise_utils.utils import get_optimizer
    from nise_utils.utils import save_checkpoint
    from nise_utils.utils import create_logger
    from simple_models.pose_resnet import get_pose_net

    def reset_config(config, args):
        if args.gpus:
            config.GPUS = args.gpus
        if args.workers:
            config.WORKERS = args.workers

        if args.simple_model_file:
            config.TEST.MODEL_FILE = args.simple_model_file

    def simple_parse_args():
        parser = argparse.ArgumentParser(description='Train keypoints network')
        # general
        parser.add_argument('--simple_cfg',
                            help='experiment configure file name',
                            required=True,
                            type=str)

        args, rest = parser.parse_known_args()
        # update config
        update_config(args.simple_cfg)

        # training
        parser.add_argument('--frequent',
                            help='frequency of logging',
                            default=simple_cfg.PRINT_FREQ,
                            type=int)
        parser.add_argument('--gpus',
                            help='gpus',
                            type=str, default='0')
        parser.add_argument('--workers',
                            help='num of dataloader workers',
                            type=int, default=8)

        parser.add_argument('--simple-model-file',
                            help='model state file',
                            type=str)

        args, rest = parser.parse_known_args()

        return args

    simple_args = simple_parse_args()
    reset_config(simple_cfg, simple_args)

    simple_human_det_model = get_pose_net(
        simple_cfg, is_train=True
    )
    gpus = [int(i) for i in simple_cfg.GPUS.split(',')]

    if simple_cfg.TEST.MODEL_FILE:
        simple_human_det_model.load_state_dict(
            torch.load(simple_cfg.TEST.MODEL_FILE))
    simple_human_det_model = torch.nn.DataParallel(
        simple_human_det_model, device_ids=gpus).cuda()
    return simple_human_det_model
# Backup
# def inference(args, epoch, data_loader, model, offset=0):
#
#     model.eval()
#
#     if args.save_flow or args.render_validation:
#         flow_folder = "{}/inference/{}.epoch-{}-flow-field".format(
#             args.save, args.name.replace('/', '.'), epoch)
#         if not os.path.exists(flow_folder):
#             os.makedirs(flow_folder)
#
#     args.inference_n_batches = np.inf if args.inference_n_batches < 0 else args.inference_n_batches
#
#     progress = tqdm(data_loader, ncols=100, total=np.minimum(len(data_loader), args.inference_n_batches),
#                     desc='Inferencing ',
#                     leave=True, position=offset)
#
#     statistics = []
#     total_loss = 0
#     for batch_idx, (data, target) in enumerate(progress):
#         if args.cuda:
#             data, target = [d.cuda(async=True) for d in data], [
#                 t.cuda(async=True) for t in target]
#         # data[0] torch.Size([8, 3, 2, 384, 1024]) ，bs x channels x num_images, h, w
#         # target torch.Size([8, 2, 384, 1024]) maybe offsets
#         data, target = [Variable(d) for d in data], [
#             Variable(t) for t in target]
#
#         # when ground-truth flows are not available for inference_dataset,
#         # the targets are set to all zeros. thus, losses are actually L1 or L2 norms of compute optical flows,
#         # depending on the type of loss norm passed in
#         with torch.no_grad():
#             # losses: list (2) output: torch.Size([bs, 2, 384, 1024])
#             losses, output = model(data[0], target[0], inference=True)
#
#         losses = [torch.mean(loss_value) for loss_value in losses]
#         loss_val = losses[0]  # Collect first loss for weight update
#         total_loss += loss_val.item()
#         loss_values = [v.item() for v in losses]
#
#         # gather loss_labels, direct return leads to recursion limit error as it looks for variables to gather'
#         loss_labels = list(model.module.loss.loss_labels)
#
#         statistics.append(loss_values)
#         # import IPython; IPython.embed()
#         if args.save_flow or args.render_validation:
#             for i in range(args.inference_batch_size):
#                 _pflow = output[i].data.cpu().numpy().transpose(1, 2, 0)
#                 flow_utils.writeFlow(join(flow_folder, '%06d.flo' % (batch_idx * args.inference_batch_size + i)),
#                                      _pflow)
#
#     progress.close()
#
#     return
