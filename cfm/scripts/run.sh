#!/bin/bash

datasets=(mnist fashion_mnist cifar10)

for ds in ${datasets[@]};
do
    channels=$(echo $ds | awk '{if ($1 == "cifar10") { a = 3 } else { a = 1 } print a}')
    epoch=$(echo $ds | awk '{if ($1 == "cifar10") { a = 10 } else { a = 10 } print a}')
    echo $ds $channels $epoch
    python ../src/train.py \
        --dataset $ds \
        --epoch $epoch \
        --channels $channels
done
