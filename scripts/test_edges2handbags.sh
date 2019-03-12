#!/usr/bin/env bash
set -ex
# models
RESULTS_DIR='./results/edges2handbags'
G_PATH='./pretrained_models/edges2handbags_net_G.pth'
E_PATH='./pretrained_models/edges2handbags_net_E.pth'

# dataset
CLASS='edges2handbags'
DIRECTION='AtoB' # from domain A to domain B
LOAD_SIZE=256 # scale images to this size
FINE_SIZE=256 # then crop to this size
INPUT_NC=1  # number of channels in the input image

# misc
GPU_ID=0   # gpu id
NUM_TEST=10 # number of input images duirng test
NUM_SAMPLES=10 # number of samples per input images

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test.py \
  --dataroot ./datasets/${CLASS} \
  --results_dir ${RESULTS_DIR} \
  --checkpoints_dir ./pretrained_models/ \
  --name ${CLASS} \
  --direction ${DIRECTION} \
  --loadSize ${FINE_SIZE} \
  --fineSize ${FINE_SIZE} \
  --input_nc ${INPUT_NC} \
  --num_test ${NUM_TEST} \
  --n_samples ${NUM_SAMPLES} \
  --center_crop \
  --no_flip

python ./test.py --dataroot ./datasets/edges2shoes --results_dir ./results/edges2shoes/zvae-nz8/test_zcode1 --checkpoints_dir ./pretrained_models --name zvae-nz8 --direction AtoB \
--loadSize 256 --fineSize 256 --input_nc 1 --num_test 200 --center_crop --no_flip --model zvae_gan