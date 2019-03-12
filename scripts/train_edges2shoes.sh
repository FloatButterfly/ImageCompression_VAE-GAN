#!/usr/bin/env bash
set -ex
MODEL='cvae_gan'
# dataset details
CLASS='edges2shoes'  # facades, day2night, edges2shoes, edges2handbags, maps
NZ=8
NO_FLIP='--no_flip'
DIRECTION='AtoB'
LOAD_SIZE=256
FINE_SIZE=256
INPUT_NC=1
NITER=30
NITER_DECAY=30

# training
GPU_ID=0
DISPLAY_ID=$((GPU_ID*10+1))
CHECKPOINTS_DIR=../checkpoints/${CLASS}/
NAME=${CLASS}_${MODEL}

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --display_id ${DISPLAY_ID} \
  --dataroot ./datasets/${CLASS} \
  --name ${NAME} \
  --model ${MODEL} \
  --direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --loadSize ${LOAD_SIZE} \
  --fineSize ${FINE_SIZE} \
  --nz ${NZ} \
  --input_nc ${INPUT_NC} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --use_dropout
python ./train.py --display_id 10 --dataroot ./datasets/edges2shoes --name edges2shoes_cvae_gan --model cvae_gan --direction AtoB --checkpoints_dir ../checkpoints/edges2shoes/ \
--loadSize 256 --fineSize 256 --nz 8 --input_nc 1 --niter 30 --niter_decay 30 --use_dropout --continue_train --epoch_count 52

python ./train.py --display_id 10 --dataroot ./datasets/edges2shoes --name edges2shoes_cvae_gan --model cvae_gan --direction AtoB --checkpoints_dir ../checkpoints/edges2shoesWithNZ16/ --loadSize 256 --fineSize 256 --nz 16 --input_nc 1 --niter 30 --niter_decay 30 --use_dropout --continue_train --epoch_count 41

python ./train.py --display_id 10 --dataroot ./datasets/edges2shoes --name edges2shoes_cvae_gan --model cvae_gan --direction AtoB --checkpoints_dir ../checkpoints/edges2shoesWithNZ32/ --loadSize 256 --fineSize 256 --nz 32 --input_nc 1 --niter 30 --niter_decay 30 --use_dropout --continue_train --epoch_count 19


##1号服务器
python ./train.py --display_id 10 --dataroot ./datasets/edges2shoes --name edges2shoes_cvae_gan --model cvae_gan --direction AtoB --checkpoints_dir ../checkpoints/edges2shoesWithNZ8/ \
--loadSize 256 --fineSize 256 --nz 8 --input_nc 1 --niter 30 --niter_decay 30 --use_dropout --continue_train --epoch_count 26

##2号服务器，zVAE,nz=8
python ./train.py --display_id 10 --dataroot ./datasets/edges2shoes --name edges2shoes_zvae_gan --model zvae_gan --direction AtoB --checkpoints_dir ../checkpoints/edges2shoesWithNZ8/ \
--loadSize 256 --fineSize 256 --nz 8 --input_nc 1 --niter 65 --niter_decay 65 --use_dropout --continue_train --epoch 115 --epoch_count 116

##handbags

python ./train.py --display_id 10 --dataroot ./datasets/edges2handbags --name edges2handbags_zvae_gan --model zvae_gan --direction AtoB --checkpoints_dir ../checkpoints/edges2handbagsWithNZ8/ \
--loadSize 256 --fineSize 256 --nz 8 --input_nc 1 --niter 60 --niter_decay 60 --use_dropout --continue_train --epoch_count 96

##handbags nz=16
python ./train.py --display_id 10 --dataroot ./datasets/edges2handbags --name edges2handbags_zvae_gan --model zvae_gan --direction AtoB --checkpoints_dir ../checkpoints/edges2handbagsWithNZ16/ \
--loadSize 256 --fineSize 256 --nz 16 --input_nc 1 --niter 60 --niter_decay 60 --use_dropout --continue_train --epoch_count 31

###1号 edges2shoes from 115
python ./train.py --display_id 10 --dataroot ./datasets/edges2shoes --name edges2shoes_zvae_gan --model zvae_gan --direction AtoB --checkpoints_dir ../checkpoints/edges2shoesWithNZ8/ \
--loadSize 256 --fineSize 256 --nz 8 --input_nc 1 --niter 110 --niter_decay 30 --use_dropout --continue_train  --epoch 115 --epoch_count 116

# handbags with lr
python ./train.py --display_id 10 --dataroot ./datasets/edges2handbags --name edges2handbags_zvae_gan --model zvae_gan --direction AtoB --checkpoints_dir ../checkpoints/edges2handbagsWithNZ8/ \
--loadSize 256 --fineSize 256 --nz 8 --input_nc 1 --niter 60 --niter_decay 60 --use_dropout --continue_train --phase train_x4 --epoch 80 --epoch_count 81

#train together
python ./train.py --display_id 10 --dataroot ./datasets/edges2handbags --model zvae_gan --direction AtoB --checkpoints_dir ../checkpoints/edges2handbagsWithNZ8/ \
--name edges2twoDatasets_zvae --loadSize 256 --fineSize 256 --nz 8 --input_nc 1 --niter 20 --niter_decay 20 --use_dropout --phase train_together --epoch 120

#labeled generator
python ./train.py --dataset_mode labeled --display_id 10 --dataroot ./datasets/together --model label_zvae --direction AtoB \
--checkpoints_dir ../checkpoints/together_labeled --name edge2shoes_and_handbags --loadSize 256 --fineSize 256 --nz 8 --input_nc 1 \
--niter 60 --niter_decay 60 --use_dropout --phase train --batch_size 4 --continue_train

#labeled encoder
python ./train.py --dataset_mode labeled --display_id 10 --dataroot ./datasets/together --model label_encoder_zvae --direction AtoB \
--checkpoints_dir ../checkpoints/together_labeled --name edge2shoes_and_handbags --loadSize 256 --fineSize 256 --nz 8 --input_nc 1 \
--niter 20 --niter_decay 20 --use_dropout --phase val