#!/bin/bash

curl -LJO https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet110-1d1ed7c2.th

mv resnet110-1d1ed7c2.th cifar_resnet110.th

curl -O http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar

curl -O https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

curl -O https://raw.githubusercontent.com/shantnu/FaceDetect/master/haarcascade_frontalface_default.xml
