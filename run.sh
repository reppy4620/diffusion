#!/bin/bash

cur=$(pwd)

cd ddpm/scripts && ./run.sh
cd $cur

cd score-sde/scripts && ./run.sh
cd $cur

cd rect_flow/scripts && ./run.sh
cd $cur
