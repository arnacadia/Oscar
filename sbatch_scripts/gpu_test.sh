#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#SBATCH --gres=gpu:1
#SBATCH --output="gpu_test.out"
#SBATCH --nodelist=torpaq

#echo "CUDA DEVICE: " $CUDA_VISIBLE_DEVICES
#echo "**** RUNNING PYGPU.TEST()"
#DEVICE="cuda" python -c "import pygpu; pygpu.test()"
echo "**** RUNNING TEST_GPU"
echo $(which python)
THEANO_FLAGS="device=cuda" python other/test_gpu.py
#echo "THEANO CONFIGURATION"
#THEANO_FLAGS="device=cuda" python -c 'import theano; print(theano.config)' | less
