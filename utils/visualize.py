import numpy as np
import visdom
from utils.imutils import batch_with_heatmap
import torch

class Visualizer(object):
    def __init__(self, env = 'default', **kwargs):
        self.vis = visdom.Visdom(env = env, **kwargs)
        self.index = {}
    
    def plot_many_stack(self, d):
        '''
        self.plot('loss',1.00)
        '''
        name = list(d.keys())
        name_total = " ".join(name)
        x = self.index.get(name_total, 0)
        val = list(d.values())
        if len(val) == 1:
            y = np.array(val)
        else:
            y = np.array(val).reshape(-1, len(val))
        # print(x)
        self.vis.line(Y = y, X = np.ones(y.shape) * x,
                      win = str(name_total),  # unicode
                      opts = dict(legend = name,
                                  title = name_total),
                      update = None if x == 0 else 'append'
                      )
        self.index[name_total] = x + 1

NUM_IMG_SHOWN_DEBUG = 2

def plt_2_viz(g):
    """
        NOT INPLACE
        img with h x w x channels to channels x h x w
    """
    
    def single(single_img):
        if single_img.shape[0] == 3:
            return single_img
        
        if type(single_img) == np.ndarray:
            g_plot = np.zeros((single_img.shape[2], single_img.shape[0], single_img.shape[1])).astype(np.float32)
            for channel in range(single_img.shape[2]):
                g_plot[channel, :, :] = single_img[:, :, channel]
        else:
            g_plot = torch.zeros((single_img.shape[2], single_img.shape[0], single_img.shape[1]), dtype = torch.float32)
            for channel in range(single_img.shape[2]):
                g_plot[channel, :, :] = single_img[:, :, channel]
        return g_plot
    
    if len(g.shape) == 4:
        # a batch
        if type(g) == np.ndarray:
            g_plot = np.zeros_like(g)
        else:
            g_plot = torch.zeros_like(g)
        
        for i in range(g.shape[0]):
            g_plot[i, ...] = single(g[i, ...])
    else:
        g_plot = single(g)
    return g_plot


def viz_plot_gt_and_pred(viz, imgs, gt_heatmaps, predicted_heatmaps):
    '''

    :param viz: Visdom variable
    :param imgs: bs x 3 x h x w, presumably. But can convert them use plt_2_viz
    :param gt_heatmaps: Must give this pls
    :param predicted_heatmaps: set it to None so only gt_heatmaps are shown
    :return:
    '''
    if gt_heatmaps is not None:
        # imgs = plt_2_viz(imgs)
        if torch.cuda.is_available():
            mean = torch.Tensor([0.5, 0.5, 0.5]).cuda()  # used in batch_with_heapmap
        else:
            mean = torch.Tensor([0.5, 0.5, 0.5])  # used in batch_with_heapmap
        gt_batch_img = batch_with_heatmap(imgs[:NUM_IMG_SHOWN_DEBUG, ...],
                                          gt_heatmaps[:NUM_IMG_SHOWN_DEBUG, ...], mean = mean,
                                          num_rows = 2)
        g_viz = plt_2_viz(gt_batch_img)
        viz.image(g_viz)
        
        if predicted_heatmaps is not None:
            predicted_heatmaps_ = predicted_heatmaps.detach()
            pred_batch_img = batch_with_heatmap(imgs[:NUM_IMG_SHOWN_DEBUG, ...],
                                                predicted_heatmaps_[:NUM_IMG_SHOWN_DEBUG, ...], mean = mean,
                                                num_rows = 2)
            p_viz = plt_2_viz(pred_batch_img)
            viz.image(p_viz)


if __name__ == '__main__':
    
    from torchnet import meter
    import time, random
    
    # 用 torchnet来存放损失函数，如果没有，请安装conda install torchnet
    '''
    训练前的模型、损失函数设置
    vis = Visualizer(env='my_wind')#为了可视化增加的内容
    loss_meter = meter.AverageValueMeter()#为了可视化增加的内容
    for epoch in range(10):
        #每个epoch开始前，将存放的loss清除，重新开始记录
        loss_meter.reset()#为了可视化增加的内容
        model.train()
        for ii,(data,label)in enumerate(trainloader):
            ...
            out=model(input)
            loss=...
            loss_meter.add(loss.data[0])#为了可视化增加的内容
    
        #loss可视化
        #loss_meter.value()[0]返回存放的loss的均值
        vis.plot_many_stack({'train_loss': loss_meter.value()[0]})#为了可视化增加的内容
    '''
    # 示例
    vis = Visualizer(env = 'my_wind')  # 为了可视化增加的内容
    loss_meter = meter.AverageValueMeter()  # 为了可视化增加的内容
    for epoch in range(103):
        time.sleep(.1)
        # loss_meter.reset()  # 为了可视化增加的内容
        loss_meter.add(epoch * random.random())  # 假设loss=epoch
        vis.plot_many_stack({'train_loss': loss_meter.value()[0]})  # 为了可视化增加的内容
        # 如果还想同时显示test loss，如法炮制,并用字典的形式赋值，如下。还可以同时显示train和test accuracy
        # vis.plot_many_stack({'train_loss': loss_meter.value()[0]，'test_loss':test_loss_meter.value()[0]})#为了可视化增加的内容
