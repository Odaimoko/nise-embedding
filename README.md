# nise-embedding



# TODOLIST

+ [x] 接口问题
+ [x] 固定 gt 的 bbox，est joints（2018-12-03）
+ [ ] ~~找到 single frame 的问题，查看输出的图像。~~
+ [x] 2018-12-06：joint 最终分数=human detection*joint
+ [ ] 训练之前 freeze 一下
+ [x] flow 用 pad 达到 32的倍数

  + [ ] （其他的方式）
+ [ ] joint prop 的 score 怎么确？
    + [ ] 单纯使用上一帧过来的 score
    + [ ] 上一帧的 score*joint 的 score
+ [ ] 达到pt17 task 3 的state-of-the-art
+ [x] 使用data parallel 加速，现在太慢了。

  + [ ] 2018-12-11：看了时间之后，发现并没有什么可以减少时间的？可能在 jointprop 和 est的时候整个batch 一起走。

  + [ ] Detected 8 boxes.发现主要是存储图像花时间了。
    人检测…… 1.136 s.
  	 	生成flow…… 0.121 s.
  	 	Proped 5 boxes
  	 	Joint prop…… 0.512 s.
  	 	Before NMS: 13 people. 	After NMS: 8 people
  	 	NMS…… 0.000 s.
  	 	关节预测…… 0.483 s.
  	 	ID分配…… 0.001 s

  + [ ] detect可以parallel？好像不行。

  + [ ] flow可以？flow 的 model 可以接受 batch 的输入，大小为 $bs\times channels\times 2\times h\times w$。但如果要这一步并行化，就要加载进所有的图片？或者也可以设置一个生成 flow 的 batchsize， 毕竟这个是 root。$2$指的是 flow 需要两张图片，如果要并行就需要$[[1,2],[2,3],[3,4]]$。
+ [x] 宁可花多一点时间去做正确，也不要回头来 debug。
+ [x] 为什么高那么多？66.7->69.6。[2018-12-12](2018-12-12)。
+ [ ] flow 的正确性怎么看？小数据集的正确性先确保了。
+ [x] 多线程。bash 多个 GPU 利用。
+ [x] 参数与设定分开。[2018-12-12](2018-12-12)。
+ [x] 存储 detection 结果
+ [ ] 存储 flow 结果
+ [ ] 只用有标注的进行 prop（四个连续的 flow 加起来作为整体 flow，比较smooth）。
    + [ ] 因为并不是所有都有标注。如果前一帧没有gt 那就用的是 det 的 box，降低精确度。

# experiment

Single-frame pose estimation training log. 

The person is cropped using gt, and the  accuracy is calculated according to single person.

person detector: [Detectron](https://github.com/roytseng-tw/Detectron.pytorch#supported-network-modules), config `my_e2e_mask_rcnn_X-101-64x4d-FPN_1x`, which has the highest boxes AP in the general fasterRCNN/Mask RCNN family. The modification is only to turn off mask. 

```
$ diff my_e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml ../Detectron.pytorch/tron_configs/baselines/e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml
5c5
<   MASK_ON: False # turn off mask
---
>   MASK_ON: True
```





## 2018-12-18

### baseline

仅仅使用 people detector 的时候，box 的 AP。

需要 nms。nms 两个 thres 分别是 0.05和0.5 。

```
valid_task_1_DETbox_allBox_propAll_propDET_nmsThres_0.05_0.5
[INFO] - In total 22807 predictions, 18966 gts.
[INFO] - AP: tensor(0.5362)
```



### 实验

prop 的时候不 filter。注意并不是所有图都有 annotation，所以虽然说是 gtpose， 有一部分仍然是 det pose。

- [ ] 【暂时不做】prop 的时候使用全部 gt 的 pose。

- [x] prop 的时候使用重合度高的 gtpose。

    - [x] score 使用原 box score * gtpose 里关节 visible 的平均值。

- [x] prop 的时候使用人检测和关节检测的结果。

    - [x] score 使用原 box score * 新 joints 的平均值。

    - [ ] ```
        valid_task_-1_DETbox_allBox_propAll_propDET_nmsThres_0.05_0.5
        [INFO] - In total 23196 predictions, 18966 gts.
        [INFO] - AP: tensor(0.5408)
        ```




### 计算 AP 的步骤

1. 在每一张图里确定 pred 是否 tp。
2. 把所有图里的pred和 gt 收集起来，gt 的总数就是总共的 positive 数量。
3. 把所有的 pred 按照 confidence排序，然后仿照voc里的。



DONE。

## 2018-12-17

讨论结果

- 做一个控制变量的实验，要做到底

今天要做的事情

- [x] 整理算法详细结果。[link](./current-alg.md).
- [x] 存储flow的结果，以及在detection结果上的estimation数据。用处是flow debug。
- [ ] 弄清楚box ap怎么计算。
- [ ] 做实验的时候目的和步骤写清楚，记下来。
- [ ] 保持每天目的记录的习惯，而不是想什么做什么。



### flow debug的流程

==标记有参数的地方==。

- [ ] 计算所有 detbox 的 AP。
- [x] 计算 flow，存储。
- [ ] prop box（过滤否，==thres==），score。==nms==。
- [ ] 记录 nms 结果的 box
- [ ] est joints