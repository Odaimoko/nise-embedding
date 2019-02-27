

# nise-embedding

My baseline

Task 1. `valid_task_1_DETbox_propfiltered_propthres_0.5_propGT`. This is obtained by not nms the detected box (i.e. use all detected box to estimate joints).

虽然样子上写着 propGT 啥的但是由于是 task1，并没有 prop 这一步，我把它取出来吧，名字改成了

​	`baseline/69.6-noProp-valid_task_1_onlyDETbox_noNMS`

```
& Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total\\
& 79.8 & 78.5 & 70.7 & 59.2 & 70.1 & 65.5 & 58.3 & 69.6 \\
```

Use GT box to estimate。(meaningless)

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
    + [x] 上一帧的 score*joint 的 score
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
+ [x] 存储 flow 结果
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



### 2019-02-27

时隔两个月再次开工……

### Something I want to check

- [x] The correctness of matching algorithm。I think this can be check by gt box matching. If gt box is used, the score of each gt box is 1 and of each gt joint is 1. So all gt joints and boxes will all be kept during filtering. As shown in the experiment in 2018-12-27, and the json directory is `pred_json-track-developing/valid_task_-2_GTbox_allBox_tfIoU_nmsThres_0.35_0.70_IoUMetric_mkrs`.

Following experiments aim to keep mAP the same and see what will MOTA be. Since they are only run for task one, no id is assigned. Experiments must be run again with task 3.

- [ ] without joint filtering: Task 3 performance of my baseline for task 1
- [ ] without joint filtering: Task 3 performance of NMSed baseline (thres are 0.35_0.50 respectively)
- [ ] with joint filtering:  T3 of the two mentioned above

Those three to-checks are actually unnecessary. I have conducted experiments about box and joint thres during the process of assigning ID. Only boxes with scores over `box_thres` have ID, and only joint with scores over `joint_thres` are output. This experiment is just a verification of what Detect-and-Track has said, that there is a tradeoff between MOTA and mAP. Filtering boxes results in lower mAP and higher MOTA.

### What  is the problem



## 2018-12-27



### 实验

使用gtbox来matching，以及使用gtbox对应gtjoint输出。当然是有filter box和joint的，但是gtbox的score都是1，而且gtjoint的score是可见度，所以不影响。这说明了matching的极限。没有满级的原因是毕竟人还是会有重叠，用box肯定会有错判。

由于使用的是gt，那些没有标注的就跳过了。

```
2018-12-27-gtmatching.txt
& MOTA & MOTA & MOTA & MOTA & MOTA & MOTA & MOTA & MOTA & MOTP & Prec & Rec  \\
& Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total& Total& Total& Total\\
& 91.0 & 91.9 & 92.5 & 92.6 & 92.3 & 92.2 & 92.1 & 92.0 & 96.8 & 99.9 & 98.3 \\
```

进行了调参，结果与2018-12-26实验里2018-12-26-filter box and joint差不了太多。

## 2018-12-26

进入tracking part。论文里有两个 filter， 首先 filter 人，还要把 filtered 人的 joint 滤一遍。

```
We first drop low-confidence detections, which tends to decrease the mAP metric but increase the MOTA tracking metric. 
Also, since the tracking metric MOT penalizes false positives equally regardless of the scores, we drop low confidence joints first to generate the result as in [11]. 
We choose the boxes and joints drop threshold in a data-driven manner on validation set, 0.5 and 0.4 respectively.
```

在 Detect-and-track 里

```
Before linking the detections in time, we drop the low-conﬁdence and potentially incorrect detections.
```

### 目标

- [x] 跑出一个总体的mota 结果，因为我现在做的肯定有问题，所以需要一步一个脚印修正。
  - [ ] 搞清楚mota在joint里怎么算
    - [x] 和mAP不同的是，tracking需要correspondence。detection里，如果两个proposal和gt重合度都>0.3，那么这两个都是tp；其他任何<=0.3的都是fp。这里就允许两个proposal和gt都有correspondence，但是tracking只有一个。
    - [x] [verified] 那我觉得tracking里有两个步骤，一个是correspondence的判定，一个是用threshold判断这个correspondence是不是valid. 
    - [x] evaluateTracking里的`computeMetrics`接受的参数是已经计算好的correspondence，直接计算mota等东西的。
  - [x] 可能是estimation的问题
      - [x] 总体的 mAP 差不多，一般不可能是 est 的问题。
  - [x] 可能是matching问题。
  - [x] 使用 gtbox 作为 matching 的来源，如果正确的话应该能够百分百追踪？
  - [x] 从1744等没有人消失并且人比较容易分辨的来看， matching 没有问题。有的人的标注只有一个关节，那这个就构不成一个 gtbox我会删除，也就没法利用 box 的 iou 进行匹配，所以可能出现 miss；因此在 estimation 的部分也会丢失这几个点。但这不是 matching，而是获取 gt 数据的问题。
  - [x] 可能是输出问题。由于如果输出的格式有问题，那么前面怎么找都不可能正确，所以先看这个。
    - [x] 确定了不是这个问题。
    - [x] 输出人物的顺序不同并不会导致mota的变化。
  - [ ] 可能是 filter 的问题。
- [x] 到底需不需要对那些不连续的帧来 tracking 。中间的部分才有 gt，才能够 matching。应该是要的，因为虽然在 gt 里隔开了，但是 id 还在。
    - [ ] 这个时候就应该出现四帧跳跃估计了？
- [x] 整理 Assign ID 的算法流程。
- [ ] 纠正 greedy matching 的错误。

### 实验

采用nmsthreshold-0.35-0.5，在assignID的时候会以0.5 filter掉人box，并没有filter掉joint。

```

& Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total\\
& 82.7 & 80.8 & 72.8 & 61.1 & 71.4 & 66.8 & 59.2 & 71.5 \\

& MOTA & MOTA & MOTA & MOTA & MOTA & MOTA & MOTA & MOTA & MOTP & Prec & Rec  \\
& Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total& Total& Total& Total\\
& 37.5 & 36.6 &  1.2 &-22.1 & 16.3 & -2.7 &-34.4 &  6.8 & 82.7 & 54.4 & 79.6 \\
```

filter掉了joint

```
2018-12-26-filter box and joint
& Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total\\
& 78.9 & 76.6 & 67.8 & 54.4 & 65.6 & 60.9 & 53.6 & 66.3 \\

& MOTA & MOTA & MOTA & MOTA & MOTA & MOTA & MOTA & MOTA & MOTP & Prec & Rec  \\
& Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total& Total& Total& Total\\
& 63.6 & 62.0 & 48.6 & 38.0 & 53.3 & 48.8 & 38.0 & 51.2 & 83.6 & 80.3 & 71.1 \\
```



只有01001的话

```
& Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total\\
& 99.9 & 98.0 & 94.8 & 89.8 & 98.7 & 95.6 & 96.2 & 96.4 \\

& MOTA & MOTA & MOTA & MOTA & MOTA & MOTA & MOTA & MOTA & MOTP & Prec & Rec  \\
& Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total& Total& Total& Total\\
& 81.0 & 77.5 & 67.7 & 62.8 & 73.9 & 76.3 & 73.8 & 73.8 & 87.0 & 81.0 & 96.9 \\
```

filter掉了joint

```
& Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total\\
& 99.9 & 98.0 & 93.2 & 87.4 & 97.9 & 93.1 & 95.0 & 95.2 \\

& MOTA & MOTA & MOTA & MOTA & MOTA & MOTA & MOTA & MOTA & MOTP & Prec & Rec  \\
& Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total& Total& Total& Total\\
& 96.9 & 93.7 & 84.9 & 79.1 & 94.7 & 88.3 & 90.1 & 90.2 & 87.2 & 94.7 & 95.6 \\
```



## 2018-12-22

### 目标

弄清楚是哪个的作用：nms 还是 flow。需要控制变量，应该先跑一次仅仅使用 detbox 并改变nms 的 thres 的实验。

看起来只是 nms 的作用

### 实验

实验结果：[link](./nms-exp.xlsx)。

- baseline-nonms：不使用 nms，直接用 detection 的 box 来 estimate。
- Detection: 只使用 detection box，加入 nms。

|                | Head | Shou | Elb  | Wri  | Hip  | Knee | Ankl | Total | preds | AP     | delta-baseline | delta-nms |
| -------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ----- | ------ | -------------- | --------- |
| baseline-nonms | 79.8 | 78.5 | 70.7 | 59.2 | 70.1 | 65.5 | 58.3 | 69.6  |       |        | 0              |           |
| Detection      |      |      |      |      |      |      |      |       |       |        |                |           |
| 0.05_0.30      | 83.4 | 80.8 | 72.7 | 61.3 | 70.4 | 65.7 | 58.1 | 71.2  | 36694 | 0.7264 | 1.6            |           |
| 0.05_0.50      | 80.3 | 79   | 71.1 | 59.6 | 70.5 | 65.8 | 58.5 | 70    | 56920 | 0.7656 | 0.4            |           |
| 0.05_0.70      | 80.3 | 79   | 71.1 | 59.6 | 70.5 | 65.8 | 58.5 | 70    | 56920 | 0.7656 | 0.4            |           |
| 0.15_0.30      | 83.1 | 80.4 | 72.5 | 61.1 | 70   | 65.5 | 58   | 71    | 29745 | 0.7241 | 1.4            |           |
| 0.15_0.50      | 82.1 | 80.5 | 72.4 | 60.6 | 71.5 | 66.8 | 59.3 | 71.2  | 39763 | 0.7647 | 1.6            |           |
| 0.15_0.70      | 82.1 | 80.5 | 72.4 | 60.6 | 71.5 | 66.8 | 59.3 | 71.2  | 39763 | 0.7647 | 1.6            |           |
| 0.25_0.30      | 82.7 | 80.1 | 72.4 | 61   | 69.7 | 65.4 | 57.9 | 70.7  | 26915 | 0.7223 | 1.1            |           |
| 0.25_0.50      | 82.7 | 80.8 | 72.7 | 60.9 | 71.6 | 67   | 59.4 | 71.5  | 33318 | 0.762  | 1.9            |           |
| 0.25_0.70      | 82.7 | 80.8 | 72.7 | 60.9 | 71.6 | 67   | 59.4 | 71.5  | 33318 | 0.762  | 1.9            |           |
| 0.35_0.30      | 82.2 | 79.7 | 72.1 | 60.8 | 69.5 | 65.3 | 57.8 | 70.5  | 24995 | 0.7205 | 0.9            |           |
| 0.35_0.50      | 83   | 80.9 | 72.8 | 61.1 | 71.6 | 66.9 | 59.4 | 71.6  | 29480 | 0.7591 | 2              |           |
| 0.35_0.70      | 83   | 80.9 | 72.8 | 61.1 | 71.6 | 66.9 | 59.4 | 71.6  | 29480 | 0.7591 | 2              |           |
| 0.45_0.30      | 81.9 | 79.4 | 72   | 60.7 | 69.4 | 65.2 | 57.8 | 70.3  | 23476 | 0.7176 | 0.7            |           |
| 0.45_0.50      | 83   | 80.9 | 72.8 | 61.1 | 71.4 | 66.8 | 59.3 | 71.6  | 26763 | 0.7539 | 2              |           |
| 0.45_0.70      | 83   | 80.9 | 72.8 | 61.1 | 71.4 | 66.8 | 59.3 | 71.6  | 26763 | 0.7539 | 2              |           |
|                |      |      |      |      |      |      |      |       |       |        |                |           |
| propGT         |      |      |      |      |      |      |      |       |       |        |                |           |
| 0.05_0.30      | 83.9 | 81.5 | 73.3 | 61.6 | 71.2 | 66.6 | 59.4 | 71.9  | 37415 | 0.7439 | 2.3            | 0.7       |
| 0.05_0.50      | 80.7 | 79.4 | 71.6 | 59.9 | 71.1 | 66.6 | 59.5 | 70.6  | 58440 | 0.7841 | 1              | 0.6       |
| 0.05_0.70      | 74.6 | 74.1 | 67.4 | 56.1 | 66.9 | 63.2 | 56.4 | 66.1  | 65294 | 0.6694 | -3.5           | -3.9      |
| 0.15_0.30      | 83.6 | 81.2 | 73.1 | 61.5 | 70.9 | 66.5 | 59.4 | 71.7  | 30241 | 0.7403 | 2.1            | 0.7       |
| 0.15_0.50      | 82.4 | 80.9 | 72.7 | 60.8 | 72   | 67.4 | 60.3 | 71.7  | 40884 | 0.7847 | 2.1            | 0.5       |
| 0.15_0.70      | 76.3 | 75.6 | 68.6 | 57.1 | 67.8 | 64.1 | 57.2 | 67.3  | 45973 | 0.6785 | -2.3           | -3.9      |
| 0.25_0.30      | 83.3 | 80.9 | 73   | 61.3 | 70.6 | 66.3 | 59.3 | 71.5  | 27323 | 0.7376 | 1.9            | 0.8       |
| 0.25_0.50      | 82.8 | 81.1 | 73   | 61.1 | 72.2 | 67.6 | 60.4 | 71.9  | 34236 | 0.7819 | 2.3            | 0.4       |
| 0.25_0.70      | 76.9 | 76.1 | 69   | 57.4 | 68   | 64.3 | 57.4 | 67.7  | 38455 | 0.6833 | -1.9           | -3.8      |
| 0.35_0.30      | 82.9 | 80.5 | 72.8 | 61.2 | 70.5 | 66.2 | 59.2 | 71.3  | 25335 | 0.7353 | 1.7            | 0.8       |
| 0.35_0.50      | 83.1 | 81.2 | 73.1 | 61.2 | 72.2 | 67.6 | 60.4 | 72    | 30224 | 0.7777 | 2.4            | 0.4       |
| 0.35_0.70      | 77.4 | 76.3 | 69.3 | 57.6 | 68.1 | 64.3 | 57.5 | 67.9  | 33757 | 0.6848 | -1.7           | -3.7      |
| 0.45_0.30      | 82.5 | 80.2 | 72.6 | 61.1 | 70.3 | 66.1 | 59.1 | 71.1  | 23799 | 0.7327 | 1.5            | 0.8       |
| 0.45_0.50      | 83.3 | 81.2 | 73.1 | 61.3 | 72   | 67.5 | 60.3 | 72    | 27390 | 0.7721 | 2.4            | 0.4       |
| 0.45_0.70      | 77.9 | 76.6 | 69.5 | 57.8 | 68.2 | 64.3 | 57.5 | 68.1  | 30301 | 0.6892 | -1.5           | -3.5      |
|                |      |      |      |      |      |      |      |       |       |        |                |           |
| propDET        |      |      |      |      |      |      |      |       |       |        |                |           |
| 0.05_0.30      | 83.6 | 81   | 72.8 | 61.4 | 70.6 | 65.8 | 58.2 | 71.4  | 36694 | 0.7264 | 1.8            | 0.2       |
| 0.05_0.50      | 80   | 78.8 | 71   | 59.4 | 70.4 | 65.7 | 58.5 | 69.8  | 56920 | 0.7656 | 0.2            | -0.2      |
| 0.05_0.70      | 75.3 | 74.8 | 67.8 | 56.4 | 67.6 | 63.4 | 56.2 | 66.5  | 56920 | 0.7656 | -3.1           | -3.5      |
| 0.15_0.30      | 83.3 | 80.7 | 72.6 | 61.2 | 70.2 | 65.6 | 58.2 | 71.1  | 29745 | 0.7241 | 1.5            | 0.1       |
| 0.15_0.50      | 81.9 | 80.3 | 72.3 | 60.5 | 71.5 | 66.7 | 59.3 | 71.1  | 39763 | 0.7647 | 1.5            | -0.1      |
| 0.15_0.70      | 77.4 | 76.5 | 69.2 | 57.6 | 68.6 | 64.5 | 57.1 | 68    | 39763 | 0.7647 | -1.6           | -3.2      |
| 0.25_0.30      | 82.8 | 80.3 | 72.5 | 61.1 | 69.9 | 65.5 | 58.1 | 70.9  | 26915 | 0.7223 | 1.3            | 0.2       |
| 0.25_0.50      | 82.5 | 80.7 | 72.6 | 60.8 | 71.6 | 66.9 | 59.4 | 71.4  | 33318 | 0.762  | 1.8            | -0.1      |
| 0.25_0.70      | 78.3 | 77.1 | 69.7 | 58.1 | 68.9 | 64.8 | 57.3 | 68.5  | 33318 | 0.762  | -1.1           | -3        |
| 0.35_0.30      | 82.4 | 79.8 | 72.3 | 60.9 | 69.7 | 65.4 | 58   | 70.6  | 24995 | 0.7205 | 1              | 0.1       |
| 0.35_0.50      | 82.8 | 80.8 | 72.8 | 60.9 | 71.6 | 66.9 | 59.3 | 71.5  | 29480 | 0.7591 | 1.9            | -0.1      |
| 0.35_0.70      | 79   | 77.7 | 70.1 | 58.4 | 69.1 | 64.9 | 57.3 | 68.8  | 29480 | 0.7591 | -0.8           | -2.8      |
| 0.45_0.30      | 81.9 | 79.5 | 72.1 | 60.8 | 69.5 | 65.3 | 57.9 | 70.4  | 23476 | 0.7176 | 0.8            | 0.1       |
| 0.45_0.50      | 82.9 | 80.7 | 72.7 | 61   | 71.3 | 66.8 | 59.2 | 71.5  | 26763 | 0.7539 | 1.9            | -0.1      |
| 0.45_0.70      | 79.5 | 78   | 70.5 | 58.7 | 69   | 64.8 | 57.3 | 69    | 26763 | 0.7539 | -0.6           | -2.6      |


## 2018-12-21

### 目标

+ [ ] ~~全部的cfg 和logger传入。~~
+ [x] 整理实验数据，选出合适的 nms 参数，并用之于 DETBOX 看效果。
    + [x] 全部都比baseline 还要低。哪里有出问题了？【[Done](Questions.md/#2018-12-21)】
    + [x] 现在看来是输出的问题，输出成了`self.people_ids`个数，实际应该是`unified_box`。判断条件那里没有加`task==-1`，改一下吧。。。
    + [x] 2018-12-21 10:51:17的程序：验证是否输出那里条件判断的问题。
    + [x] 并不是。
+ [x] 有两个选项，一是直接用 unifiedbox作为`run_one_video_flow_debug`的 detect_box，不加 flow 跑。第二个是相当于重新用 flow 算一遍。我不知道第一个能不能兼容多GPU， 因为我多 GPU 的目的是载入足够多的 flow。~~暂时试试第一个。~~重新计算，因为之前的 joint est 有误，所以 box 并不准确。
+ [ ] 试着阅读`tracking_engine_DAT`（应该是 Detect-and-Track，3d mask-rcnn 的那个）。



### 实验结果







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
    Imoko:py oda$  /usr/local/bin/python2.7 /Users/oda/posetrack/poseval/py/evaluate.py --groundTruth=val_gt_task1/ --predictions=valid_task_-1_DETbox_allBox_propAll_propGT_tfIoU_nmsThres_0.05_0.5/ --evalPoseEstimation
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