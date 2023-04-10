#!/bin/bash

datasets=(mnist)

for ds in ${datasets[@]};
do
    config_path=../configs/$ds.yaml
    python ../src/train.py --config $config_path
done
