#!/usr/bin/env python
# coding: utf-8

#pip install --upgrade pip
#conda install -c conda-forge cudatoolkit=11.8.0
#pip install opencv-python matplotlib imageio gdown
#pip install cuda-python==11.8.0
#pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
#pip install tensorrt

#Test1___________________________________________________________
#module swap PrgEnv-cray PrgEnv-nvidia/8.3.2
#module load cseinit-noloads
#module load cse/anaconda3/latest
#pip install opencv-python imageio gdown
#pip install cuda-python==11.8.0
#pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
#pip install tensorrt
#conda install -c conda-forge cudatoolkit=11.8.0

#Test2___________________________________________________________

import tensorflow as tf
from tensorflow import keras

print(f'\nTensorflow version = {tf.__version__}\n')
print("CheckPoint 1!!!")
print(f'\nNumber of GPUs = {tf.config.list_physical_devices("GPU")}\n')


