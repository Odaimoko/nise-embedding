#!/usr/bin/env bash
for t1 in $(seq 0.05 .1 .3)
do
    for t2 in $(seq .3 .1 .7)
    do
        echo Running posetrack 17: NMS thresholds are $t1, $t2.
        bash run-all-workers-4.sh $t1 $t2
        echo NMS thresholds $t1, $t2 have been tested.
    done
done
