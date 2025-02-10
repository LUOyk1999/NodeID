python main-arxiv.py --dataset ogbn-arxiv --hidden_channels 256 --epochs 1000 --lr 0.0005 --runs 1 --local_layers 5 --post_bn --device 7


python arxiv_ID_MLP.py --lr 0.01 --hidden_channels 256 --num_layers 4 \
            --epochs 2000 --device 7 --dropout 0.5 --num_id 15 --k 0 --norm_type batch --runs 2
