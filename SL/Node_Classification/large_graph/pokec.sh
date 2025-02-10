python main-batch.py --dataset pokec --hidden_channels 256 --epochs 2000 --batch_size 550000 --lr 0.0005 --runs 1 --local_layers 7 --in_drop 0.0 --dropout 0.2 --weight_decay 0.0 --post_bn --eval_step 9 --eval_epoch 1000 --device 0

python ID_MLP.py --dataset pokec --lr 0.001 --hidden_channels 256 --num_layers 5 \
            --epochs 2000 --device 1 --dropout 0.5 --num_id 21 --k 0 --norm_type batch --runs 2 --eval_step 9 --eval_epoch 100
