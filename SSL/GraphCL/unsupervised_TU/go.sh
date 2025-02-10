#!/bin/bash -ex

for layers in 3 4 5 6
do
for num_code in 16 8 4 32 20
do
for seed in 0 1 2
do
  CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --num_code $num_code --DS $2 --lr 0.01 --local --num-gc-layers $layers --aug random2 --seed $seed

done
done
done

# dataset layers codebook_size
# DD 4 4
# NCI1 5 4
# PROTEINS 3 8
# COLLAB 5 32
# IMDB-B 3 8
# RDT-B 5 4
# RDT-M5K 4 4
# MUTAG 4 16