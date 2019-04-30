cuda_all=export CUDA_VISIBLE_DEVICES=0,1,2,3
cuda_0=export CUDA_VISIBLE_DEVICES=0
cuda_1=export CUDA_VISIBLE_DEVICES=1
mot_pypath=export PYTHONPATH=../poseval/py-motmetrics:$${PYTHONPATH}
nise_main=python scripts/run.py
nise_main_mnet=python scripts/run-mnet.py

# cd
cd_deep=cd ../deep-pt
cd_simple=cd ../simple-baseline-pytorch

tron_cfg_mask=--tron_cfg exp_config/detectron/my_e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml --load_detectron ~/zhangxt/disk/pretrained/e2e_mask_rcnn_X-101-64x4d-FPN_1x.pkl  --dataset coco2017
tron_cfg_faster=--tron_cfg ../Detectron.forpt/tron_configs/baselines/e2e_faster_rcnn_X-101-64x4d-FPN_1x.yaml --load_detectron ~/zhangxt/disk/pretrained/e2e_faster_rcnn_X-101-64x4d-FPN_1x.pkl  --dataset coco2017

hrcfg=--hr-cfg ../deep-pt/experiments/pt17/hrnet-coco-w48_384x288-from-freeze.yaml --hr-model ../deep-pt/output-freeze/pt17/pose_hrnet/hrnet-coco-w48_384x288-from-freeze/pt17-epoch-20-90.54428065311858
hr_90472=--hr-cfg ../deep-pt/experiments/pt17/hrnet-coco-w48_384x288-from-freeze-colorrgbFalse.yaml --hr-model ../deep-pt/output-freeze/pt17/pose_hrnet/hrnet-coco-w48_384x288-from-freeze-colorrgbFalse/pt17-epoch-20-90.47223881413662
sb_90=--simple_cfg ../simple-baseline-pytorch/experiments/pt17/res152-coco-384x288.yaml  --simple-model-file /root/zhangxt/disk/posetrack/simple-baseline-pytorch/output-pt17-fromfreeze/pt17/pose_resnet_152/res152-coco-384x288/pt17-epoch-20-90.04363546829477
sb_88=--simple_cfg ../simple-baseline-pytorch/experiments/pt17/res50-coco-256x192_d256x3_adam_lr1e-3.yaml  --simple-model-file ../simple-baseline-pytorch/output-pt17-freeze/pt17/pose_resnet_50/res50-coco-256x192_d256x3_adam_lr1e-3/pt17-epoch-16-88.01324110762047
flow_cfg=--model FlowNet2S --flownet_resume ../flownet2-pytorch/FlowNet2-S_checkpoint.pth.tar

eval_gt_debug=-g pred_json-pre-commissioning/val_gt_task3-debugging/
eval_gt_all=-g pred_json-pre-commissioning/val_gt_task1/

nise_1_nmson=--nise_config exp_config/1/t-nmsON+flip.yaml
nise_1_nmson_train=--nise_config exp_config/1/t-train-nmsON+flip.yaml
nise_1_nmson_faster=--nise_config exp_config/1/t-nmsON+flip-faster.yaml
nise_1_gtbox=--nise_config exp_config/1/t-gt+flip.yaml
nise_1_gtjoints=--nise_config exp_config/1/t-nmsON+flip-gtjoints.yaml
nise_1_nmson_debug=--nise_config exp_config/1/t-nmsON+flip-debug.yaml
nise_1_gtbox_debug=--nise_config exp_config/1/t-gt+flip-debug.yaml


nise_2_root=--nise_config exp_config/2/t-flow-root.yaml

nise_3_root=--nise_config exp_config/3/t-3-root.yaml
nise_3_root_fb=--nise_config exp_config/3/t-3-root-fb.yaml
nise_3_root_nofbj=--nise_config exp_config/3/t-3-root-nofbj.yaml
nise_3_root_vis=--nise_config exp_config/3/t-3-root-vis.yaml
nise_3_gen_matched_detbox=--nise_config exp_config/3/t-1-matched_detbox.yaml
nise_3_gen_matched_detbox_gtid=--nise_config exp_config/3/t-1-matched_detbox-gtid.yaml
nise_3_gen_matched_detbox_gtid_fj=--nise_config exp_config/3/t-1-matched_detbox-gtid-fj.yaml
nise_3_gen_matched_detbox_gtid_vis=--nise_config exp_config/3/t-1-matched_detbox-gtid-vis.yaml
nise_3_gen_matched_detbox_hipckh=--nise_config exp_config/3/t-1-matched_detbox-gtid-hi_pckh.yaml
nise_3_gen_matched_detbox_hipckh_fj=--nise_config exp_config/3/t-1-matched_detbox-gtid-hi_pckh-fj.yaml
nise_3_matched_detbox=--nise_config exp_config/3/t-3-matched_detbox.yaml
nise_3_matched_detbox_vis=--nise_config exp_config/3/t-3-matched_detbox-vis.yaml
nise_3_matched_detbox_fb=--nise_config exp_config/3/t-3-matched_detbox-fb.yaml
nise_3_matched_detbox_fj=--nise_config exp_config/3/t-3-matched_detbox-fj.yaml
nise_3_matched_detbox_fbj=--nise_config exp_config/3/t-3-matched_detbox-fbj.yaml
nise_3_matched_detbox_hi=--nise_config exp_config/3/t-3-matched_detbox-hi.yaml
nise_3_matched_detbox_hi_fj=--nise_config exp_config/3/t-3-matched_detbox-hi-fj.yaml

nise_gen_fmap=--nise_config exp_config/others/gen_fmap.yaml
nise_gen_fmap_train=--nise_config exp_config/others/gen_fmap-train.yaml
nise_gen_training_set=--nise_config exp_config/others/gen_training_set.yaml


train_mNet=--nise_config exp_config/train_mNet/train.yaml
train_mNet_mr=--nise_config exp_config/train_mNet/train_with_maskRCNN.yaml

# train
train-mNet-debug:
	 $(cuda_all);python -mpdb scripts/train_matchingNet.py  $(tron_cfg_mask) $(train_mNet)
train-mNet:
	 $(cuda_all);python scripts/train_matchingNet.py  $(tron_cfg_mask) $(train_mNet)
train-mNet-mr:
	 $(cuda_all);python scripts/train_matchingNet.py  $(tron_cfg_mask) $(train_mNet_mr)


# task2
t2-sb88:
	$(cuda_all); $(nise_main) $(flow_cfg) $(sb_88) $(tron_cfg_mask) $(nise_2_root)

# MNET tracking
nise_3_root_mnet=--nise_config exp_config/3/t-3-root-mnet.yaml
nise_3_root_mnet_greedy=--nise_config exp_config/3/t-3-root-mnet-greedy.yaml

t3-sb88-mnet:
	$(nise_main_mnet)  $(tron_cfg_mask) --task1pred pred_json-single-est/79.0-sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50 $(nise_3_root_mnet)
t3-sb88-mnet-greedy:
	$(nise_main_mnet)  $(tron_cfg_mask) --task1pred pred_json-single-est/79.0-sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50 $(nise_3_root_mnet_greedy)


eval-t3-sb88-mnet-fbj:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-track-mnet/valid_task_-6_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.80_joint_0.50_matchID_posThres_0/ --evalPoseEstimation --evalPoseTracking


eval-t3-sb88-mnet-debug-fbj:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_debug) -p  pred_json-track-mnet/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.80_joint_0.50_matchID/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-mnet-greedy-debug:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_debug) -p  pred_json-track-mnet/valid_task_-6_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_greedy_box_0.80_joint_0.50_matchID/ --evalPoseTracking

nise_3_greedy=--nise_config exp_config/3/t-3-root-greedy.yaml
t3-sb88-greedy:
	$(nise_main)  $(tron_cfg_mask) --task1pred pred_json-single-est/79.0-sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50 $(nise_3_greedy)

eval-t3-sb88-greedy:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-track/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_greedy_box_0.80_joint_0.50_matchID/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-greedy-debug:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_debug) -p  pred_json-track/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_greedy_box_0.80_joint_0.50_matchID/ --evalPoseEstimation --evalPoseTracking


# BOX tracking
t3-hr904-nms-.35-.5-boxjoint-.8-.5:
	$(nise_main) --task1pred pred_json-single-est/81.9-hr904-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50 $(nise_3_root)

t3-sb88-nms-.35-.5-boxjoint-.8-.5:
	$(nise_main) --task1pred pred_json-single-est/79.0-sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50 $(nise_3_root)

t3-sb88-nms-.35-.5-boxjoint-0-0:
	$(nise_main) --task1pred pred_json-single-est/79.0-sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50 $(nise_3_root_nofbj)

t3-sb88-nms-.35-.5-boxjoint-.8-0:
	$(nise_main) --task1pred pred_json-single-est/79.0-sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50 $(nise_3_root_fb)


t3-sb88-gen-matched_detbox:
	$(nise_main) --task1pred pred_json-single-est/79.0-sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50 $(nise_3_gen_matched_detbox)

t3-sb88-gen-matched_detbox-gtid:
	$(nise_main) --task1pred pred_json-single-est/79.0-sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50 $(nise_3_gen_matched_detbox_gtid)

t3-sb88-gen-matched_detbox-gtid-fj:
	$(nise_main) --task1pred pred_json-single-est/79.0-sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50 $(nise_3_gen_matched_detbox_gtid_fj)


t3-sb88-gen-matched_detbox-gtid-hi_pckh:
	$(nise_main) --task1pred pred_json-single-est/79.0-sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50 $(nise_3_gen_matched_detbox_hipckh)

t3-sb88-gen-matched_detbox-gtid-hi_pckh-fj:
	$(nise_main) --task1pred pred_json-single-est/79.0-sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50 $(nise_3_gen_matched_detbox_hipckh_fj)


t3-sb88-matched_detbox:
	$(nise_main) $(nise_3_matched_detbox)

t3-sb88-matched_detbox-fb:
	$(nise_main) $(nise_3_matched_detbox_fb)
t3-sb88-matched_detbox-fj:
	$(nise_main) $(nise_3_matched_detbox_fj)
t3-sb88-matched_detbox-fbj:
	$(nise_main) $(nise_3_matched_detbox_fbj)

t3-sb88-gen-matched_detbox-gtid-vis:
	$(nise_main) --task1pred pred_json-single-est/79.0-sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50 $(nise_3_gen_matched_detbox_gtid_vis)

t3-sb88-matched_detbox-vis:
	$(nise_main) $(nise_3_matched_detbox_vis)
t3-sb88-nms-.35-.5-boxjoint-.8-.5-vis:
	$(nise_main) --task1pred pred_json-single-est/79.0-sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50 $(nise_3_root_vis)


t3-sb88-matched_detbox-hi:
	$(nise_main) $(nise_3_matched_detbox_hi)
t3-sb88-matched_detbox-hi-fj:
	$(nise_main) $(nise_3_matched_detbox_hi_fj)


nise_3_gen_matched_joints=--nise_config exp_config/3/t-1-matched_joints.yaml
nise_3_gen_mb_mj=--nise_config exp_config/3/t-1-mb+mj.yaml
nise_3_gen_mb_mj_gtid=--nise_config exp_config/3/t-1-mb+mj-gtid.yaml

nise_3_matched_joints=--nise_config exp_config/3/t-3-mj.yaml
nise_3_mj_fb=--nise_config exp_config/3/t-3-mj-fb.yaml
nise_3_mj_mb=--nise_config exp_config/3/t-3-mj-mb.yaml
nise_3_mj_mb_gtid=--nise_config exp_config/3/t-1-mj+mb-gtid.yaml

t3-sb88-gen_matched_joints:
	$(nise_main) $(nise_3_gen_matched_joints)

t3-sb88-gen_mb+mj:
	$(nise_main) $(nise_3_gen_mb_mj)

t3-sb88-gen_mb+mj-gtid:
	$(nise_main) $(nise_3_gen_mb_mj_gtid)

t3-sb88-matched_joints:
	$(nise_main) $(nise_3_matched_joints)

t3-sb88-mj-fb:
	$(nise_main) $(nise_3_mj_fb)

t3-sb88-mj-mb:
	$(nise_main) $(nise_3_mj_mb)
t3-sb88-mj-mb-gtid:
	$(nise_main) $(nise_3_mj_mb_gtid)

eval-t3-sb88-gen_matched_joints: #t3-sb88-gen_matched_joints
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-matchedJoints/valid_task_-4_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.00_joint_0.01_matchID/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-gen_mb_mj: t3-sb88-gen_mb+mj
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-matchedJoints/valid_task_-3_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.00_joint_0.01_matchID_lessFP/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-mj: t3-sb88-matched_joints
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-matchedJoints/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.00_joint_0.01_matchID/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-mj-fb: t3-sb88-mj-fb
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-matchedJoints/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.50_joint_0.01_matchID/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-mj-mb:#t3-sb88-mj-mb
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-matchedJoints/valid_task_-2_mj-mb/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-mj-mb-gtid: #t3-sb88-mj-mb-gtid
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-matchedJoints/valid_task_-3_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.00_joint_0.01_gtID_lessFP/ --evalPoseEstimation --evalPoseTracking



# sb
sb-val-mpii:
	$(cd_simple); python pose_estimation/valid.py \
	--cfg experiments/mpii/resnet101/384x384_d256x3_adam_lr1e-3.yaml \
    --flip-test \
    --model-file models/pose_mpii/pose_resnet_101_384x384.pth.tar

sb-val-88:
	$(cd_simple);$(cuda_all); python pose_estimation/valid_for_pt.py --cfg experiments/pt17/res50-coco-256x192_d256x3_adam_lr1e-3.yaml  --model-file output-pt17-freeze/pt17/pose_resnet_50/res50-coco-256x192_d256x3_adam_lr1e-3/pt17-epoch-16-88.01324110762047


# hr
hr-test-90.544:
	$(cd_deep); python tools/test-pt.py --cfg experiments/pt17/hrnet-coco-w48_384x288-from-freeze.yaml TEST.MODEL_FILE output-freeze/pt17/pose_hrnet/hrnet-coco-w48_384x288-from-freeze/pt17-epoch-20-90.54428065311858
hr-train-freeze-w48-384:
	$(cd_deep); python tools/train-pt-freeze-1.py --cfg experiments/pt17/hrnet-coco-w48_384x288-freeze-1-colorrgbFalse.yaml TEST.MODEL_FILE models/pose_coco/pose_hrnet_w48_384x288.pth
hr-train-finetune-w48-384:
	$(cd_deep); python tools/train-pt-from-freeze.py --cfg experiments/pt17/hrnet-coco-w48_384x288-from-freeze-colorrgbFalse.yaml TEST.MODEL_FILE output-freeze/pt17/pose_hrnet/hrnet-coco-w48_384x288-freeze-1-colorrgbFalse/pt17-epoch-1-43.33863767970564
hr-test-90.472:
	$(cd_deep);$(cuda_all); python tools/test-pt.py --cfg experiments/pt17/hrnet-coco-w48_384x288-from-freeze-colorrgbFalse.yaml TEST.MODEL_FILE output-freeze/pt17/pose_hrnet/hrnet-coco-w48_384x288-from-freeze-colorrgbFalse/pt17-epoch-20-90.47223881413662

# task1
t1-sb-88-train:
	$(cuda_0); $(nise_main) $(flow_cfg) $(sb_88) $(tron_cfg_mask) $(nise_1_nmson_train)
t1-sb-88:
	$(cuda_all); $(nise_main) $(flow_cfg) $(sb_88) $(tron_cfg_mask) $(nise_1_nmson)
t1-sb-88-debug:
	$(cuda_0); python -mpdb scripts/run.py $(flow_cfg) $(sb_88) $(tron_cfg_mask) $(nise_1_nmson_debug)
t1-sb-88-gtbox:
	$(cuda_all); $(nise_main) $(flow_cfg) $(sb_88) $(tron_cfg_mask) $(nise_1_gtbox)


t1-sb-90:
	$(cuda_all); $(nise_main) $(flow_cfg) $(sb_90) $(tron_cfg_mask) $(nise_1_nmson)

t1-hr-90.544:
	$(cuda_all); $(nise_main) $(flow_cfg) $(hrcfg) $(tron_cfg_mask) $(nise_1_nmson)

t1-hr-90.472:
	$(cuda_all); $(nise_main) $(flow_cfg) $(hr_90472) $(tron_cfg_mask) $(nise_1_nmson)


t1-hr-90.544-debug:
	$(cuda_all); $(nise_main) $(flow_cfg) $(hrcfg) $(tron_cfg_mask) $(nise_1_nmson_debug)

t1-hr-90.472-debug:
	$(cuda_all); $(nise_main) $(flow_cfg) $(hr_90472) $(tron_cfg_mask) $(nise_1_nmson_debug)

t1-hr-90.472-gtbox:
	$(cuda_all); $(nise_main) $(flow_cfg) $(hr_90472) $(tron_cfg_mask) $(nise_1_gtbox)

t1-sb-88-gtbox-debug:
	$(cuda_all); $(nise_main) $(flow_cfg) $(sb_88) $(tron_cfg_mask) $(nise_1_gtbox_debug)
t1-hr-90.472-gtbox-debug:
	$(cuda_all); $(nise_main) $(flow_cfg) $(hr_90472) $(tron_cfg_mask) $(nise_1_gtbox_debug)

t1-faster-sb-88:
	$(cuda_0); $(nise_main) $(flow_cfg) $(sb_88) $(tron_cfg_faster) $(nise_1_nmson_faster)

t1-gen_fmap:
	$(cuda_0); $(nise_main) $(tron_cfg_mask) $(nise_gen_fmap)
t1-gen_fmap-train:
	$(cuda_0); $(nise_main) $(tron_cfg_mask) $(nise_gen_fmap_train)

t1-gen_training_set:
	$(cuda_0); $(nise_main) $(tron_cfg_mask) $(nise_gen_training_set)


#commissioning
commi-bjthres:
	python scripts/param_box_joint.py --nise_config exp_config/3/t-3-root-commi-box-joint.yaml

commi-bjthres-88:
	python scripts/param_box_joint.py --nise_config exp_config/3/t-3-88-commi-box-joint.yaml
commi-eval-bjthres-hr904:
	$(mot_pypath); python scripts/task3-batch-eval.py --root pred_json-track-bjparam-hr904

commi-eval-bjthres-sb88:
	$(mot_pypath); python scripts/task3-batch-eval.py --root pred_json-track-bjparam-sb88




# eval

eval-t1-sb-88:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p pred_json-single-est/sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50/ --evalPoseEstimation -o out-single-est
eval-t1-hr-904:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p pred_json-single-est/hr904-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50/ --evalPoseEstimation -o out-single-est
eval-t1-faster:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p pred_json-single-est/valid_task_1_faster_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50/ --evalPoseEstimation -o out-single-est

eval-t1-sb-88-debug:
	$(mot_pypath); python ../poseval/py/evaluate.py  $(eval_gt_debug) -p pred_json-single-est/79.0-sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50/ --evalPoseEstimation -o out-single-est
eval-t1-hr-905-debug:
	$(mot_pypath); python ../poseval/py/evaluate.py  $(eval_gt_debug) -p pred_json-single-est/hr905-valid_task_1_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50/ --evalPoseEstimation -o out-single-est
eval-t1-hr-904-debug:
	$(mot_pypath); python ../poseval/py/evaluate.py  $(eval_gt_debug) -p pred_json-single-est/valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50/ --evalPoseEstimation -o out-single-est


eval-t2-sb88-debug:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_debug) -p  pred_json-flow/valid_task_-1_mask_DETbox_allBox_noFlip_estJoints_propAll_propDET_tfIoU_nmsThres_0.35_0.50/ --evalPoseEstimation --evalPoseTracking


eval-t3-hr904:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-track/hr904-64.0-valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.80_joint_0.50_matchID/ --evalPoseEstimation --evalPoseTracking -o out-hr904-tracking


eval-t3-sb88-t1json:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-single-est/79.0-sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50/ --evalPoseEstimation --evalPoseTracking -o out-hr904-tracking > evalt3-sb88-t1json

eval-t3-sb88-gen_matched_box:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-matchedDet/valid_task_-3_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-gen_matched_box-gtid:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-matchedDet/valid_task_-3_gtid/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-gen_matched_box-gtid-fj:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-matchedDet/valid_task_-3_gtid-fj/ --evalPoseEstimation --evalPoseTracking


eval-t3-sb88-gen_matched_box-gtid-debug:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_debug) -p  pred_json-matchedDet/valid_task_-3_gtid/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-gen_matched_box-hipckh-debug:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_debug) -p  pred_json-matchedDet-hipckh/valid_task_-3_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_hiPckh_0.8/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-normal:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-track/sb88-60.1-normal-task_-2_box_0.80_joint_0.50_matchID/ --evalPoseEstimation --evalPoseTracking


eval-t3-sb88-nofbj:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-track/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.00_joint_0.00_matchID/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-normal-debug:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_debug) -p  pred_json-track/sb88-60.1-normal-task_-2_box_0.80_joint_0.50_matchID/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-matched_box:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-matchedDet/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.00_joint_0.00_matchID/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-matched_box-fb:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-matchedDet/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.80_joint_0.00_matchID/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-matched_box-fj:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-matchedDet/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.00_joint_0.50_matchID/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-matched_box-fbj:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-matchedDet/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.80_joint_0.50_matchID/ --evalPoseEstimation --evalPoseTracking



eval-t3-sb88-matched_box-debug:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_debug) -p  pred_json-matchedDet/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.00_joint_0.00_matchID/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-matched_box-debug-fb:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_debug) -p  pred_json-matchedDet/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.80_joint_0.00_matchID/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-matched_box-debug-fj:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_debug) -p  pred_json-matchedDet/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.00_joint_0.50_matchID/ --evalPoseEstimation --evalPoseTracking


eval-t3-sb88-matched_box-debug-fbj:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_debug) -p  pred_json-matchedDet/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.80_joint_0.50_matchID/ --evalPoseEstimation --evalPoseTracking


