#!/bin/bash

datasets=("huggan/AFHQv2" mnist fashion_mnist cifar10)

for ds in ${datasets[@]};
do
    config_path=../configs/$ds.yaml
    echo $config_path
    python ../src/train.py --config $config_path
done
