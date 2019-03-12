#!/usr/bin/env bash
set -ex
# models
RESULTS_DIR='./results/edges2shoes'
G_PATH='./pretrained_models/edges2shoes_net_G.pth'
E_PATH='./pretrained_models/edges2shoes_net_E.pth'

# dataset
CLASS='edges2shoes'
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

# cvae-nz8
python ./test.py --dataroot ./datasets/edges2shoes --results_dir ./results/edges2shoes/cvae-nz8 --checkpoints_dir ./pretrained_models --name cvae-nz8 --direction AtoB \
--loadSize 256 --fineSize 256 --input_nc 1 --num_test 10 --center_crop --no_flip

#cvae-nz16
python ./test.py --dataroot ./datasets/edges2shoes --results_dir ./results/edges2shoes/cvae-nz16 --checkpoints_dir ./pretrained_models --name cvae-nz16 --direction AtoB \
--loadSize 256 --fineSize 256 --input_nc 1 --num_test 200 --center_crop --no_flip --nz 16 --test_psnr --test_ssim --test_msssim

#zvae-nz8
python ./test.py --dataroot ./datasets/edges2shoes --results_dir ./results/edges2shoes/zvae-nz8/epoch120_test_original_binary --checkpoints_dir ./pretrained_models --name zvae-nz8 --direction AtoB \
--loadSize 256 --fineSize 256 --input_nc 1 --num_test 200 --center_crop --no_flip --phase val_original_binary
#
python ./test.py --dataroot ./datasets/edges2shoes --results_dir ./results/edges2shoes/zvae-nz8/test_zcode1 --checkpoints_dir ./pretrained_models --name zvae-nz8 --direction AtoB \
--loadSize 256 --fineSize 256 --input_nc 1 --num_test 200 --center_crop --no_flip --model zvae_gan

#random
python ./test.py --dataroot ./datasets/edges2shoes --results_dir ./results/edges2shoes/zvae-nz8/epoch120_random --checkpoints_dir ./pretrained_models --name zvae-nz8 --direction AtoB \
--loadSize 256 --fineSize 256 --input_nc 1 --num_test 200 --center_crop --no_flip --model zvae_gan --n_samples 10

##variable
python ./test_variable.py --dataroot ./datasets/edges2shoes --results_dir ./results/edges2shoes/zvae-nz8/epoch120_variable --checkpoints_dir ./pretrained_models --name zvae-nz8 --direction AtoB \
--loadSize 256 --fineSize 256 --input_nc 1 --num_test 30 --center_crop --no_flip --model zvae_gan

##test_handbags epoch65
python ./test.py --dataroot ./datasets/edges2handbags --results_dir ./results/edges2handbags/zvae-nz8/epoch120 --checkpoints_dir ./pretrained_models/edges2handbags/ --name zvae-nz8 --direction AtoB \
--loadSize 256 --fineSize 256 --input_nc 1 --num_test 200 --center_crop --no_flip --model zvae_gan  --epoch 120

#test shoes downsample
python ./test.py --dataroot ./datasets/edges2shoes --results_dir ./results/edges2shoes/zvae-nz8/epoch120_test_downsample_qp32_x4 --checkpoints_dir ./pretrained_models/edges2shoes --name zvae-nz8 --direction AtoB \
--loadSize 256 --fineSize 256 --input_nc 1 --num_test 200 --center_crop --no_flip --model zvae_gan --phase val_qp32 --epoch 120


python ./test.py --dataroot ./datasets/edges2shoes --results_dir ./results/edges2handbags/zvae-nz8/epoch120_val_upsample_x4_qp35_dbpn_binary  --checkpoints_dir ./pretrained_models/edges2handbags --name zvae-nz8 --direction AtoB \
--loadSize 256 --fineSize 256 --input_nc 1 --num_test 200 --center_crop --no_flip --model zvae_gan --epoch 120 --phase val_upsample_x4_qp35_dbpn_binary

python ./test.py --dataroot ./datasets/edges2shoes --results_dir ./results/edges2shoes/zvae-nz8/epoch120_val_upsample_x4_qp35_dbpn_binary  --checkpoints_dir ./pretrained_models/edges2shoes --name zvae-nz8 --direction AtoB \
--loadSize 256 --fineSize 256 --input_nc 1 --num_test 200 --center_crop --no_flip --model zvae_gan --epoch 120 --phase val_upsample_x4_qp35_dbpn_binary

##test_handbags train_v4
python ./test.py --dataroot ./datasets/edges2handbags --results_dir ./results/edges2handbags/zvae-nz8/train_x4_110 --checkpoints_dir ./pretrained_models/edges2handbags/ --name zvae-nz8 --direction AtoB \
--loadSize 256 --fineSize 256 --input_nc 1 --num_test 600 --center_crop --no_flip --model zvae_gan  --epoch 110 --phase train_x4

##test_handbags val
python ./test.py --dataroot ./datasets/edges2handbags --results_dir ./results/edges2handbags/zvae-nz8/val_x4_90 --checkpoints_dir ./pretrained_models/edges2handbags/ --name zvae-nz8 --direction AtoB \
--loadSize 256 --fineSize 256 --input_nc 1 --num_test 200 --center_crop --no_flip --model zvae_gan  --epoch 90 --phase val

# cross validation
python ./test.py --dataroot ./datasets/edges2shoes --results_dir ./results/edges2shoes/zvae-nz8/hbModelTest --checkpoints_dir ./pretrained_models/edges2handbags/ --name zvae-nz8 --direction AtoB \
--loadSize 256 --fineSize 256 --input_nc 1 --num_test 200 --center_crop --no_flip --model zvae_gan  --epoch 120 --phase val

#test
python ./test.py --dataroot ./datasets/edges2handbags --results_dir ./results/edges2handbags/zvae-nz8/together --checkpoints_dir ./pretrained_models/edges2handbags/ --name zvae-bigger --direction AtoB \
--loadSize 256 --fineSize 256 --input_nc 1 --num_test 200 --center_crop --no_flip --model zvae_gan  --epoch 35 --phase val

python ./test.py --dataroot ./datasets/edges2shoes --results_dir ./results/edges2handbags/zvae-nz8/shoes_together --checkpoints_dir ./pretrained_models/edges2handbags/ --name zvae-bigger --direction AtoB \
--loadSize 256 --fineSize 256 --input_nc 1 --num_test 200 --center_crop --no_flip --model zvae_gan  --epoch 35 --phase val

#test_latent
python ./test_latent.py --dataroot ./datasets/edges2handbags --results_dir ./results/edges2handbags/zvae-nz8/epoch120_latent --checkpoints_dir ./pretrained_models/edges2handbags/ --name zvae-nz8 --direction AtoB \
--loadSize 256 --fineSize 256 --input_nc 1 --num_test 10 --center_crop --no_flip --model zvae_gan --epoch 120 --phase train











