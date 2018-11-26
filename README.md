# nise-embedding



# experiment

Single-frame pose estimation training log. The person is cropped using gt, and the  accuracy is calculated according to single person.

```
Epoch 0
2018-11-21 19:31:30,988 | Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1 |
2018-11-21 19:31:30,988 |---|---|---|---|---|---|---|---|---|---|
2018-11-21 19:31:30,988 | 256x192_pose_resnet_50_d256d256d256 | 83.295 | 91.002 | 85.800 | 79.028 | 85.300 | 82.029 | 77.540 | 84.642 | 22.017 |
........
Epoch 5
2018-11-21 21:16:21,459 | Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1 |
2018-11-21 21:16:21,459 |---|---|---|---|---|---|---|---|---|---|
2018-11-21 21:16:21,459 | 256x192_pose_resnet_50_d256d256d256 | 92.736 | 91.836 | 86.993 | 80.692 | 86.890 | 83.286 | 78.440 | 86.942 | 29.620 |
......
Epoch 15
2018-11-21 23:25:31,966 | Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1 |
2018-11-21 23:25:31,966 |---|---|---|---|---|---|---|---|---|---|
2018-11-21 23:25:31,967 | 256x192_pose_resnet_50_d256d256d256 | 92.961 | 92.084 | 87.142 | 80.501 | 86.502 | 83.242 | 78.967 | 87.027 | 30.438 |
Epoch 16
2018-11-21 23:40:34,888 | Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1 |
2018-11-21 23:40:34,888 |---|---|---|---|---|---|---|---|---|---|
2018-11-21 23:40:34,888 | 256x192_pose_resnet_50_d256d256d256 | 92.980 | 92.075 | 87.071 | 80.674 | 86.626 | 83.462 | 79.078 | 87.109 | 30.494 |
Epoch 17
2018-11-21 23:55:28,076 | Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1 |
2018-11-21 23:55:28,077 |---|---|---|---|---|---|---|---|---|---|
2018-11-21 23:55:28,077 | 256x192_pose_resnet_50_d256d256d256 | 92.974 | 92.185 | 87.184 | 80.624 | 86.569 | 83.400 | 78.928 | 87.106 | 30.567 |
Epoch 18
2018-11-22 00:11:41,637 | Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1 |
2018-11-22 00:11:41,637 |---|---|---|---|---|---|---|---|---|---|
2018-11-22 00:11:41,638 | 256x192_pose_resnet_50_d256d256d256 | 93.093 | 92.124 | 87.133 | 80.752 | 86.623 | 83.451 | 79.055 | 87.140 | 30.615 |
Epoch 19
2018-11-22 00:27:51,210 | Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1 |
2018-11-22 00:27:51,211 |---|---|---|---|---|---|---|---|---|---|
2018-11-22 00:27:51,211 | 256x192_pose_resnet_50_d256d256d256 | 93.126 | 92.207 | 87.140 | 80.737 | 86.607 | 83.415 | 79.079 | 87.151 | 30.606 |
```



Using Model from epoch 18, use it on pt17 validation for PT task 2 & 3 (multi-frame pose est and tracking)

```
& Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total\\
& 12.6 & 11.8 &  7.1 &  4.1 &  7.7 &  3.8 &  1.8 &  7.4 \\
```

For task 1

```

Namespace(evalPoseEstimation=True, evalPoseTracking=False, groundTruth='../../nise_embedding/pred_json/val_gt_task1/', outputDir='./out', predictions='../../nise_embedding/pred_json/valid_anno_json_pred/', saveEvalPerSequence=False)
Loading data
('# gt frames  :', 66558)
('# pred frames:', 66558)
Evaluation of per-frame multi-person pose estimation
('saving results to', './out/total_AP_metrics.json')
Average Precision (AP) metric:
& Head & Shou & Elb  & Wri  & Hip  & Knee & Ankl & Total\\
& 14.0 & 13.6 &  8.8 &  5.1 & 10.2 &  5.7 &  3.3 &  9.0 \\
```



