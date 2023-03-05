#!/bin/bash

datasets=(fashion_mnist mnist cifar10)
sdes=(vp subvp)

for ds in ${datasets[@]};
do
    for sde in ${sdes[@]};
    do
        echo $ds $sde
        channels=$(echo $ds | awk '{if ($1 == "cifar10") { a = 3 } else { a = 1 } print a}')
        echo $channels
        python ../src/train.py \
            --dataset $ds \
            --sde_type $sde \
            --channels $channels
    done    
done
