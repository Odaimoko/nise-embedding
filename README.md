# nise-embedding

My baseline

Task 1. `valid_task_1_DETbox_propfiltered_propthres_0.5_propGT`. This is obtained by not nms the detected box (i.e. use all detected box to estimate joints).

```
& Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total\\
& 79.8 & 78.5 & 70.7 & 59.2 & 70.1 & 65.5 & 58.3 & 69.6 \\
```

Use GT box to estimate。

```
& Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total\\
& 85.8 & 82.3 & 73.8 & 62.6 & 73.4 & 69.5 & 64.5 & 74.0 \\
```



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



## 2018-12-21

### 目标

+ [ ] 全部的cfg 和logger传入。
+ [x] 整理实验数据，选出合适的 nms 参数，并用之于 DETBOX 看效果。
    + [ ] 全部都比baseline 还要低。哪里有出问题了？
    + [ ] 现在看来是输出的问题，输出成了`self.people_ids`个数，实际应该是`unified_box`。判断条件那里没有加`task==-1`，改一下吧。。。
    + [ ] 2018-12-21 10:51:17的程序：验证是否输出那里条件判断的问题。
+ [ ] 试着阅读`tracking_engine_DAT`（应该是 Detect-and-Track，3d mask-rcnn 的那个）。



### 实验结果

logs 文件夹里的evaluate-boxes-all文件。

```
[OUT] - 0.05_0.30
[OUT] - In total 37494 predictions.	AP: 0.7435
[OUT] - 0.05_0.50
[OUT] - In total 59101 predictions.	AP: 0.7834
Imoko:py oda$ /usr/local/bin/python2.7 /Users/oda/posetrack/poseval/py/evaluate.py --groundTruth=../../nise_embedding/pred_json/val_gt_task1/ --predictions=../../nise_embedding/pred_json/valid_task_-1_DETbox_allBox_propAll_propGT_tfIoU_nmsThres_0.05_0.50/ --evalPoseEstimation
('# gt frames  :', 2607, '# pred frames:', 2607)
& Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total\\
& 74.7 & 73.6 & 64.4 & 52.7 & 63.6 & 59.5 & 52.4 & 63.8 \\

[OUT] - 0.05_0.70
[OUT] - In total 66543 predictions.	AP: 0.6707
[OUT] - 0.05_0.90
[OUT] - In total 79194 predictions.	AP: 0.2879
[OUT] - 0.15_0.30
[OUT] - In total 30249 predictions.	AP: 0.7399
[OUT] - 0.15_0.50
[OUT] - In total 41299 predictions.	AP: 0.7843
[OUT] - 0.15_0.70
[OUT] - In total 46846 predictions.	AP: 0.6794
[OUT] - 0.15_0.90
[OUT] - In total 57524 predictions.	AP: 0.3137
[OUT] - 0.25_0.30
[OUT] - In total 27323 predictions.	AP: 0.7373
[OUT] - 0.25_0.50
[OUT] - In total 34511 predictions.	AP: 0.7818
[OUT] - 0.25_0.70
[OUT] - In total 39060 predictions.	AP: 0.6831
[OUT] - 0.25_0.90
[OUT] - In total 48673 predictions.	AP: 0.3301
[OUT] - 0.30_0.50
[OUT] - In total 25334 predictions.	AP: 0.7352
[OUT] - 0.35_0.30
[OUT] - In total 30386 predictions.	AP: 0.7774
[OUT] - 0.35_0.50
[OUT] - In total 34194 predictions.	AP: 0.6854
[OUT] - 0.35_0.70
[OUT] - In total 43030 predictions.	AP: 0.3451
[OUT] - 0.35_0.90
[OUT] - In total 23798 predictions.	AP: 0.7326
[OUT] - 0.45_0.30
[OUT] - In total 27502 predictions.	AP: 0.7719
[OUT] - 0.45_0.50
[OUT] - In total 30593 predictions.	AP: 0.6899
[OUT] - 0.45_0.70
[OUT] - In total 38591 predictions.	AP: 0.3660
```

找不回去——我他妈下次一定要备份了。

## 2018-12-20

- [x] 【昨日】调参代码，用一个py完成。
- [ ] 全部的cfg 和logger 传入。



### 

## 2018-12-19

### 目标

- [x] 调参代码，用一个py完成。

### 实验

使用tensorflow 的 iou， 和 pycocotools 对比，做三个完全相同的实验。

+ [x] 重新跑一次 detect，no nms 的 estimation 结果。

    + [x] 正确，就是我现在的最高点（69.4）

+ [x] 使用土法 nms，得到 box 的 ap

    + [x] 对task 1进行 nms。[根本就没有滤掉好吧，但是 mAP 又有提升]

    ```
    valid_task_1_DETbox_allBox_tfIoU_nmsThres_0.05_0.5
    [INFO] - In total 56920 predictions, 18536 gts.
    [INFO] - AP: tensor(0.7656)
    & Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total\\
    & 80.3 & 79.0 & 71.1 & 59.6 & 70.5 & 65.8 & 58.5 & 70.0 \\
    ```

    + [x] prop 的时候使用重合度高的 gtpose。

    ```
    valid_task_-1_DETbox_allBox_propAll_propGT_tfIoU_nmsThres_0.05_0.5
    [INFO] - In total 58440 predictions, 18536 gts.
    [INFO] - AP: tensor(0.7841)
    Imoko:py oda$  /usr/local/bin/python2.7 /Users/oda/posetrack/poseval/py/evaluate.py --groundTruth=../../nise_embedding/pred_json/val_gt_task1/ --predictions=../../nise_embedding/pred_json/valid_task_-1_DETbox_allBox_propAll_propGT_tfIoU_nmsThres_0.05_0.5/ --evalPoseEstimation
    ('# gt frames  :', 2607, '# pred frames:', 2607)
    & Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total\\
    & 80.7 & 79.4 & 71.6 & 59.9 & 71.1 & 66.6 & 59.5 & 70.6 \\
    ```

    + [x] prop 的时候使用人检测和关节检测的结果。

    ```
    valid_task_-1_DETbox_allBox_propAll_propDET_tfIoU_nmsThres_0.05_0.5
    [INFO] - In total 59101 predictions, 18536 gts.
    [INFO] - AP: tensor(0.7662)
    & Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total\\
    & 80.0 & 78.8 & 71.0 & 59.4 & 70.4 & 65.7 & 58.5 & 69.8 \\
    ```




按道理，TensorFlow 版本计算的 iou 是正确的，点应该要比 pycocotools 高。

### 实验

对 nms 的两个 threshold 调参

- `thres_1`: 0.05:0.05:0.5
- `thres_2`: 0.3:0.1:0.9











## 2018-12-18

### baseline

现在nms 的时候使用的iou 全部都是pycocotools.mask.iou，而不是使用 TensorFlow 里面的。这个感觉更不准确，但是更有效。

- [ ] 仅仅使用 people detector 的时候，box 的 AP。【待实验是否这样】

without nms

```
[INFO] - In total 56920 predictions, 18538 gts.
[INFO] - AP: tensor(0.7656)
```

需要 nms。nms 两个 thres 分别是 0.05和0.5 。

```
valid_task_1_DETbox_allBox_propAll_propDET_nmsThres_0.05_0.5
[INFO] - In total 25282 predictions, 18536 gts.
[INFO] - AP: tensor(0.5803)
```

### 实验

现在nms 的时候使用的iou 全部都是pycocotools.mask.iou，而不是使用 TensorFlow 里面的。这个感觉更不准确，但是更有效。

prop 的时候不 filter。注意并不是所有图都有 annotation，所以虽然说是 gtpose， 有一部分仍然是 det pose。

- [ ] 【暂时不做】prop 的时候使用全部 gt 的 pose。

- [x] prop 的时候使用重合度高的 gtpose。

    - [x] score 使用原 box score * gtpose 里关节 visible 的平均值。

    ```
    valid_task_-1_DETbox_allBox_propAll_propGT_nmsThres_0.05_0.5
    [INFO] - In total 23301 predictions, 18536 gts.
    [INFO] - AP: tensor(0.5734)
    ```

- [x] prop 的时候使用人检测和关节检测的结果。

    - [x] score 使用原 box score * 新 joints 的平均值。

    - [ ] ```valid_task_-1_DETbox_allBox_propAll_propDET_nmsThres_0.05_0.5
      [INFO] - In total 23154 predictions, 18536 gts.
      [INFO] - AP: tensor(0.5552)
      ```

~~过滤~~

- [ ] ~~使用重合度高的 gtpose~~
- [ ] ~~prop 的时候使用人检测和关节检测的结果。~~





### 计算 AP 的步骤

1. 在每一张图里确定 pred 是否 tp。
2. 把所有图里的pred和 gt 收集起来，gt 的总数就是总共的 positive 数量。
3. 把所有的 pred 按照 confidence排序，然后仿照voc里的。



DONE。

### 发现bug

posetrack 里，有些人的标记只有一个点。这样的面积就是0，要去掉。

## 2018-12-17

讨论结果

- 做一个控制变量的实验，要做到底

今天要做的事情

- [x] 整理算法详细结果。[link](./current-alg.md).
- [x] 存储flow的结果，以及在detection结果上的estimation数据。用处是flow debug。
- [x] 弄清楚box ap怎么计算。
- [ ] 做实验的时候目的和步骤写清楚，记下来。
- [ ] 保持每天目的记录的习惯，而不是想什么做什么。



### flow debug的流程

==标记有参数的地方==。

- [x] 计算所有 detbox 的 AP。
- [x] 计算 flow，存储。
- [x] prop box（过滤否，==thres==），score。==nms==。
- [x] 记录 nms 结果的 box
- [x] est joints