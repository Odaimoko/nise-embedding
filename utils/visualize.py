import numpy as np
import visdom


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
        loss_meter.reset()  # 为了可视化增加的内容
        loss_meter.add(epoch * random.random())  # 假设loss=epoch
        vis.plot_many_stack({'train_loss': loss_meter.value()[0]})  # 为了可视化增加的内容
        # 如果还想同时显示test loss，如法炮制,并用字典的形式赋值，如下。还可以同时显示train和test accuracy
        # vis.plot_many_stack({'train_loss': loss_meter.value()[0]，'test_loss':test_loss_meter.value()[0]})#为了可视化增加的内容
