#!/bin/bash

curl -O https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet110-1d1ed7c2.th

mv resnet110-1d1ed7c2.th util_files/cifar_resnet110.th

curl -O http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar

mv resnet18_places365.pth.tar util_files/resnet18_places365.pth.tar

curl -O https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

mv lid.176.bin util_files/lid.176.bin

curl -O https://raw.githubusercontent.com/shantnu/FaceDetect/master/haarcascade_frontalface_default.xml

mv haarcascade_frontalface_default.xml util_files/haarcascade_frontalface_default.xml