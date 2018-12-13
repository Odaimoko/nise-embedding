
from=$1
to=$2
echo $from
echo $to
export CUDA_VISIBLE_DEVICES=1
~/anaconda3/bin/python batch_run_singleGPU.py --f 3 --t 4 &
export CUDA_VISIBLE_DEVICES=2
~/anaconda3/bin/python batch_run_singleGPU.py --f 3 --t 4 &
export CUDA_VISIBLE_DEVICES=3
~/anaconda3/bin/python batch_run_singleGPU.py --f $from --t $to

