#!/bin/bash

datasets=(mnist fashion_mnist cifar10 "huggan/AFHQv2")

for ds in ${datasets[@]};
do
    config_path=../configs/$ds.yaml
    echo $config_path
    python ../src/train.py --config $config_path
done
