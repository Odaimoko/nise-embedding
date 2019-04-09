cuda_all=export CUDA_VISIBLE_DEVICES=0,1,2,3
cuda_0=export CUDA_VISIBLE_DEVICES=0
mot_pypath=export PYTHONPATH=../poseval/py-motmetrics:$${PYTHONPATH}
nise_main=python scripts/run.py

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
nise_1_nmson_faster=--nise_config exp_config/1/t-nmsON+flip-faster.yaml
nise_1_gtbox=--nise_config exp_config/1/t-gt+flip.yaml
nise_1_gtjoints=--nise_config exp_config/1/t-nmsON+flip-gtjoints.yaml
nise_1_nmson_debug=--nise_config exp_config/1/t-nmsON+flip-debug.yaml
nise_1_gtbox_debug=--nise_config exp_config/1/t-gt+flip-debug.yaml

nise_3_root=--nise_config exp_config/3/t-3-root.yaml
nise_3_root_vis=--nise_config exp_config/3/t-3-root-vis.yaml
nise_3_gen_matched_detbox=--nise_config exp_config/3/t-1-matched_detbox.yaml
nise_3_matched_detbox=--nise_config exp_config/3/t-3-matched_detbox.yaml
nise_3_matched_detbox_fb=--nise_config exp_config/3/t-3-matched_detbox-fb.yaml
nise_3_matched_detbox_fj=--nise_config exp_config/3/t-3-matched_detbox-fj.yaml
nise_3_matched_detbox_fbj=--nise_config exp_config/3/t-3-matched_detbox-fbj.yaml

#tracking
t3-hr904-nms-.35-.5-boxjoint-.8-.5:
	$(nise_main) --task1pred pred_json-single-est/81.9-hr904-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50 $(nise_3_root)

t3-sb88-nms-.35-.5-boxjoint-.8-.5:
	$(nise_main) --task1pred pred_json-single-est/79.0-sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50 $(nise_3_root)


t3-sb88-gen-oracle-matched_detbox:
	$(nise_main) --task1pred pred_json-single-est/79.0-sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50 $(nise_3_gen_matched_detbox)

t3-sb88-matched_detbox:
	$(nise_main) $(nise_3_matched_detbox)

t3-sb88-matched_detbox-fb:
	$(nise_main) $(nise_3_matched_detbox_fb)
t3-sb88-matched_detbox-fj:
	$(nise_main) $(nise_3_matched_detbox_fj)
t3-sb88-matched_detbox-fbj:
	$(nise_main) $(nise_3_matched_detbox_fbj)

t3-sb88-nms-.35-.5-boxjoint-.8-.5-vis:
	$(nise_main) --task1pred pred_json-single-est/79.0-sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50 $(nise_3_root_vis)

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
t1-sb-88:
	$(cuda_all); $(nise_main) $(flow_cfg) $(sb_88) $(tron_cfg_mask) $(nise_1_nmson)


t1-sb-90:
	$(cuda_all); $(nise_main) $(flow_cfg) $(sb_90) $(tron_cfg_mask) $(nise_1_nmson)

t1-hr-90.544:
	$(cuda_all); $(nise_main) $(flow_cfg) $(hrcfg) $(tron_cfg_mask) $(nise_1_nmson)

t1-hr-90.472:
	$(cuda_all); $(nise_main) $(flow_cfg) $(hr_90472) $(tron_cfg_mask) $(nise_1_nmson)

t1-sb-88-debug:
	$(cuda_all); $(nise_main) $(flow_cfg) $(sb_88) $(tron_cfg_mask) $(nise_1_nmson_debug)

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

eval-t3-hr904:
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-track/hr904-64.0-valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.80_joint_0.50_matchID/ --evalPoseEstimation --evalPoseTracking -o out-hr904-tracking


eval-t3-sb88-t1json: # f
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-single-est/79.0-sb88-valid_task_1_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50/ --evalPoseEstimation --evalPoseTracking -o out-hr904-tracking > evalt3-sb88-t1json

eval-t3-sb88-gen_matched_box: # f
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-track/80.8-sb88-valid_task_-3_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50/ --evalPoseEstimation --evalPoseTracking >  evalt3-sb88-gen_matched_box

eval-t3-sb88-normal: # f
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_debug) -p  pred_json-track/sb88-60.1-normal-task_-2_box_0.80_joint_0.50_matchID/ --evalPoseEstimation --evalPoseTracking



eval-t3-sb88-matched_box: # f
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-matchedDet/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.00_joint_0.00_matchID/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-matched_box-fb: # f
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-matchedDet/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.80_joint_0.00_matchID/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-matched_box-fj: # f
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-matchedDet/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.00_joint_0.50_matchID/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-matched_box-fbj: # f
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_all) -p  pred_json-matchedDet/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.80_joint_0.50_matchID/ --evalPoseEstimation --evalPoseTracking



eval-t3-sb88-matched_box-debug: # f
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_debug) -p  pred_json-matchedDet/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.00_joint_0.00_matchID/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-matched_box-debug-fb: # f
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_debug) -p  pred_json-matchedDet/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.80_joint_0.00_matchID/ --evalPoseEstimation --evalPoseTracking

eval-t3-sb88-matched_box-debug-fj: # f
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_debug) -p  pred_json-matchedDet/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.00_joint_0.50_matchID/ --evalPoseEstimation --evalPoseTracking


eval-t3-sb88-matched_box-debug-fbj: # f
	$(mot_pypath); python ../poseval/py/evaluate.py $(eval_gt_debug) -p  pred_json-matchedDet/valid_task_-2_mask_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50_IoUMetric_mkrs_box_0.80_joint_0.50_matchID/ --evalPoseEstimation --evalPoseTracking