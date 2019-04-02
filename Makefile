cuda_all=export CUDA_VISIBLE_DEVICES=0,1,2,3
eval_env=export PYTHONPATH=/root/zhangxt/disk/posetrack/poseval/py-motmetrics:$${PYTHONPATH}
nise_main=python scripts/run.py

# cd dir
cd_deep=cd ../deep-pt


tron_cfg=--tron_cfg exp_config/detectron/my_e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml --load_detectron ~/zhangxt/disk/pretrained/e2e_mask_rcnn_X-101-64x4d-FPN_1x.pkl  --dataset coco2017

hrcfg=--hr-cfg ../deep-pt/experiments/pt17/hrnet-coco-w48_384x288-from-freeze.yaml --hr-model ../deep-pt/output-freeze/pt17/pose_hrnet/hrnet-coco-w48_384x288-from-freeze/pt17-epoch-20-90.54428065311858
hr_90472=--hr-cfg ../deep-pt/experiments/pt17/hrnet-coco-w48_384x288-from-freeze-colorrgbFalse.yaml --hr-model ../deep-pt/output-freeze/pt17/pose_hrnet/hrnet-coco-w48_384x288-from-freeze-colorrgbFalse/pt17-epoch-20-90.47223881413662
sb_90=--simple_cfg ../simple-baseline-pytorch/experiments/pt17/res152-coco-384x288.yaml --gpus 0 --simple-model-file /root/zhangxt/disk/posetrack/simple-baseline-pytorch/output-pt17-fromfreeze/pt17/pose_resnet_152/res152-coco-384x288/pt17-epoch-20-90.04363546829477
sb_88=--simple_cfg ../simple-baseline-pytorch/experiments/pt17/res50-coco-256x192_d256x3_adam_lr1e-3.yaml --gpus 0 --simple-model-file ../simple-baseline-pytorch/output-pt17-freeze/pt17/pose_resnet_50/res50-coco-256x192_d256x3_adam_lr1e-3/pt17-epoch-16-88.01324110762047
flow_cfg=--model FlowNet2S --flownet_resume ../flownet2-pytorch/FlowNet2-S_checkpoint.pth.tar

nise_1_nmson=--nise_config exp_config/1/t-nmsON+flip.yaml
nise_1_gtbox=--nise_config exp_config/1/t-gt+flip.yaml
nise_1_gtjoints=--nise_config exp_config/1/t-nmsON+flip-gtjoints.yaml
nise_1_nmson_debug=--nise_config exp_config/1/t-nmsON+flip-debug.yaml
nise_1_gtbox_debug=--nise_config exp_config/1/t-gt+flip-debug.yaml

# eval

eval-t1-sb-88:
	$(eval_env); python ../poseval/py/evaluate.py -g pred_json-pre-commissioning/val_gt_task1/ -p pred_json-single-est/valid_task_1_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50/ --evalPoseEstimation -o out-single-est
eval-t1-hr-904:
	$(eval_env); python ../poseval/py/evaluate.py -g pred_json-pre-commissioning/val_gt_task1/ -p pred_json-single-est/hr904-valid_task_1_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50/ --evalPoseEstimation -o out-single-est

eval-t1-sb-88-debug:
	$(eval_env); python ../poseval/py/evaluate.py -g pred_json-pre-commissioning/val_gt_task3-debugging/ -p pred_json-single-est/sb88-valid_task_1_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50/ --evalPoseEstimation -o out-single-est
eval-t1-hr-905-debug:
	$(eval_env); python ../poseval/py/evaluate.py -g pred_json-pre-commissioning/val_gt_task3-debugging/ -p pred_json-single-est/hr905-valid_task_1_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50/ --evalPoseEstimation -o out-single-est
eval-t1-hr-904-debug:
	$(eval_env); python ../poseval/py/evaluate.py -g pred_json-pre-commissioning/val_gt_task3-debugging/ -p pred_json-single-est/hr904-valid_task_1_DETbox_allBox_Flip_estJoints_tfIoU_nmsThres_0.35_0.50/ --evalPoseEstimation -o out-single-est

# hr
hr-test-90.544:
	$(cd_deep); python tools/test-pt.py --cfg experiments/pt17/hrnet-coco-w48_384x288-from-freeze.yaml TEST.MODEL_FILE output-freeze/pt17/pose_hrnet/hrnet-coco-w48_384x288-from-freeze/pt17-epoch-20-90.54428065311858
hr-train-freeze-w48-384:
	$(cd_deep); python tools/train-pt-freeze-1.py --cfg experiments/pt17/hrnet-coco-w48_384x288-freeze-1-colorrgbFalse.yaml TEST.MODEL_FILE models/pose_coco/pose_hrnet_w48_384x288.pth
hr-train-finetune-w48-384:
	$(cd_deep); python tools/train-pt-from-freeze.py --cfg experiments/pt17/hrnet-coco-w48_384x288-from-freeze-colorrgbFalse.yaml TEST.MODEL_FILE output-freeze/pt17/pose_hrnet/hrnet-coco-w48_384x288-freeze-1-colorrgbFalse/pt17-epoch-1-43.33863767970564

# task1
t1-sb-88:
	$(cuda_all); $(nise_main) $(flow_cfg) $(sb_88) $(tron_cfg) $(nise_1_nmson)

t1-hr-90.544:
	$(cuda_all); $(nise_main) $(flow_cfg) $(hrcfg) $(tron_cfg) $(nise_1_nmson)

t1-hr-90.472:
	$(cuda_all); $(nise_main) $(flow_cfg) $(hr_90472) $(tron_cfg) $(nise_1_nmson)

t1-sb-88-debug:
	$(cuda_all); $(nise_main) $(flow_cfg) $(sb_88) $(tron_cfg) $(nise_1_nmson_debug)

t1-hr-90.544-debug:
	$(cuda_all); $(nise_main) $(flow_cfg) $(hrcfg) $(tron_cfg) $(nise_1_nmson_debug)

t1-hr-90.472-debug:
	$(cuda_all); $(nise_main) $(flow_cfg) $(hr_90472) $(tron_cfg) $(nise_1_nmson_debug)


t1-hr-90.472-gtbox-debug:
	$(cuda_all); $(nise_main) $(flow_cfg) $(hr_90472) $(tron_cfg) $(nise_1_gtbox_debug)

