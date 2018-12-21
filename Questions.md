 # Questions.md

- [ ] 改了一些参数，在某些视频上能够提升 mAP， 其他的没有效果。那么我是不是需要从整体来看效果（跑整个数据集）？只在小的 set 上实验可能错过在全部数据集上有效的参数。
- [ ] 影响 mAP 的因素太多，怎么判定是哪里出问题了？



## 2018-12-21

终极目标：回到70.6。什么因素会导致变化，1box2est。box 看起来差不多为什么会得出迥异的 est？

61.6版本（最垃圾的）和70.6版本的区别在于，有相同的 unified box，但是不一样的 estimation 结果。

63.6的会少一些estimation，但是 box 不变？

总结：三个人的 box 都一样。问题来了，你 joint 不一样，怎么会有相同的 box 呢？

两种可能，现在的 box 导致70.6的结果，或者61.6的结果。使用现在的box 来 estimation 一下？

```
全部的，为了备份谁是谁

pred_json-pre-commissioning/valid_task_-1_DETbox_allBox_propAll_propGT_tfIoU_nmsThres_0.05_0.5-70.6

# gt frames  : 2607 # pred frames: 2607
& Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total\\
& 80.7 & 79.4 & 71.6 & 59.9 & 71.1 & 66.6 & 59.5 & 70.6 \\


--predictions=pred_json-debug/valid_task_-1_DETbox_allBox_propAll_propGT_tfIoU_nmsThres_0.05_0.50/ --evalPoseEstimation
# gt frames  : 2607 # pred frames: 2607
& Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total\\
& 71.7 & 71.2 & 62.5 & 51.0 & 62.1 & 57.7 & 50.6 & 61.7 \\
```





先专注验证一下342

```
70.6
[OUT] - Pkl saved. Evaluate box AP unifed_boxes-pre-commissioning/valid_task_-1_DETbox_allBox_propAll_propGT_tfIoU_nmsThres_0.05_0.5
[OUT] - In total 413 predictions, 96 gts.
[OUT] - AP: 0.5487

# gt frames  : 48 # pred frames: 48
& Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total\\
& 39.8 & 39.2 & 19.4 & 14.9 & 35.3 & 23.1 & 12.9 & 27.3 \\


61.6
[OUT] - Pkl saved. Evaluate box AP unifed_boxes-debug/valid_task_-1_DETbox_allBox_propAll_propGT_tfIoU_nmsThres_0.05_0.50
[OUT] - In total 416 predictions, 96 gts.
[OUT] - AP: 0.5489

# gt frames  : 48 # pred frames: 48
& Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total\\
& 50.4 & 54.3 & 25.4 & 19.5 & 28.8 & 52.2 & 29.8 & 38.1 \\
```

wait 让我们先分个支，因为这个居然可以提高这么多（gt 是45.2）

