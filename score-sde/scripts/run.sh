#!/bin/bash

datasets=(cifar10)
sdes=(vp subvp)

for ds in ${datasets[@]};
do
    for sde in ${sdes[@]};
    do
        channels=$(echo $ds | awk '{if ($1 == "cifar10") { a = 3 } else { a = 1 } print a}')
        epoch=$(echo $ds | awk '{if ($1 == "cifar10") { a = 50 } else { a = 20 } print a}')
        echo $ds $sde $channels $epoch
        python ../src/train.py \
            --dataset $ds \
            --sde_type $sde \
            --epoch $epoch \
            --channels $channels
    done    
done
