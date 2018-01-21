sudo cp cuda/include/cudnn.h /usr/local/cuda/include/

sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/

sudo chmod a+r /usr/local/cuda/include/cudnn.h

sudo chmod a+r /usr/local/cuda/lib64/libcudnn*

sudo ldconfig

cuda 版本
cat /usr/local/cuda/version.txt

cudnn 版本
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2



#!/usr/bin/env sh

set -e

t=$(date +%Y-%m-%d_%H:%M:%S)

LOG=./6.0_resnet50_$t.log

../../build/tools/caffe time -model ResNet-50-deploy.prototxt -iterations 10 -gpu 0 2>&1 | tee $LOG$@
