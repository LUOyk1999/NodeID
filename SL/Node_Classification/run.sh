# GAT
python main.py --dataset coauthor-cs --hidden_channels 64 --epochs 1600 --lr 0.001 --runs 1 --local_layers 7 --weight_decay 5e-4 --dropout 0.5 --in_dropout 0.3 --num_heads 8 --device 4 --save_model --num_codes 16  --kmeans 1

python ID_MLP.py --dataset coauthor-cs --lr 0.001 --num_layers 4 \
            --hidden_channels 512 --weight_decay 5e-5 --dropout 0.5 \
            --norm_type batch --device 4 --runs 3 --num_id 21 --k 0 --epochs 2000

python main.py --dataset amazon-photo --hidden_channels 64 --epochs 1200 --lr 0.001 --runs 1 --local_layers 6 --weight_decay 5e-5 --dropout 0.7 --in_dropout 0.2 --num_heads 8 --device 2 --save_model --num_codes 4 --kmeans 1

python ID_MLP.py --dataset amazon-photo --lr 0.001 --num_layers 4 \
            --hidden_channels 256 --weight_decay 5e-5 --dropout 0.5 \
            --norm_type batch --device 2 --runs 3 --num_id 18 --k 0 --epochs 2000

python main.py --dataset coauthor-physics --hidden_channels 32 --epochs 1600 --lr 0.001 --runs 1 --local_layers 5 --weight_decay 5e-4 --dropout 0.5 --in_dropout 0.3 --num_heads 8 --device 1 --save_model --num_codes 4  --kmeans 1

python ID_MLP.py --dataset coauthor-physics --lr 0.001 --num_layers 4 \
            --hidden_channels 1024 --weight_decay 5e-5 --dropout 0.5 \
            --norm_type batch --device 1 --runs 3 --num_id 15 --epochs 2000

python main.py --dataset wikics --hidden_channels 512 --epochs 1100 --lr 0.001 --runs 1 --local_layers 8 --weight_decay 0.0 --dropout 0.5 --in_dropout 0.5 --num_heads 1 --device 2 --save_model --num_codes 8  --kmeans 1

python ID_MLP.py --dataset wikics --lr 0.001 --num_layers 4 \
            --hidden_channels 256 --weight_decay 5e-5 --dropout 0.5 \
            --norm_type batch --device 2 --runs 3 --num_id 24 --epochs 2000

python main.py --dataset amazon-ratings --hidden_channels 256 --epochs 2700 --lr 0.001 --runs 1 --local_layers 12 --weight_decay 0.0 --dropout 0.3 --in_dropout 0.2 --num_heads 2 --device 3 --save_model  --num_codes 16  --kmeans 1

python ID_MLP.py --dataset amazon-ratings --lr 0.001 --num_layers 4 \
        --hidden_channels 256 --weight_decay 5e-5 --dropout 0.5 \
        --norm_type batch --device 3 --runs 3 --num_id 36 --epochs 2000

python -u main.py  --dataset questions --hidden_channels 64 --epochs 1500 --lr 3e-5 --runs 1 --local_layers 5 --weight_decay 0.0 --dropout 0.2 --num_heads 8 --metric rocauc --device 0 --in_dropout 0.15 --pre_ln --num_codes 4 --kmeans 1

python ID_MLP.py --dataset questions --lr 0.001 --num_layers 4 \
            --hidden_channels 256 --weight_decay 5e-5 --dropout 0.5 \
            --norm_type batch --device 0 --runs 3 --num_id 15 --epochs 2000

python main.py --dataset amazon-computer --hidden_channels 64 --epochs 1200 --lr 0.001 --runs 1 --local_layers 6 --weight_decay 5e-5 --dropout 0.7 --in_dropout 0.2 --num_heads 8 --device 4 --save_model --num_codes 8  --kmeans 1

python ID_MLP.py --dataset amazon-computer --lr 0.001 --num_layers 5 \
            --hidden_channels 256 --weight_decay 5e-5 --dropout 0.5 \
            --norm_type batch --device 4 --runs 3 --num_id 18 --epochs 2000

python main.py --dataset cora --lr 0.005 --local_layers 4 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.0 --method gat --rand_split --seed 123 --device 2 --runs 1 --num_codes 8 --epoch 1000 --kmeans 1

python ID_MLP.py --dataset cora --lr 0.001 --num_layers 2 \
        --hidden_channels 512 --weight_decay 5e-4 --dropout 0.5 \
        --rand_split \
        --seed 123 --device 2 --runs 5 --num_id 12 --k 0 --epoch 1000

python main.py --dataset citeseer --lr 0.005 --local_layers 4 --hidden_channels 64 --weight_decay 0.01 --dropout 0.5 --method gat --rand_split --seed 123 --device 3 --runs 1 --num_codes 8 --epoch 1000 --kmeans 1

python ID_MLP.py --dataset citeseer --lr 0.001 --num_layers 5 \
        --hidden_channels 256 --weight_decay 0.01 --dropout 0.5 \
        --rand_split \
        --seed 123 --device 3 --runs 5 --num_id 12 --k 0 --epoch 1000

python main.py --dataset pubmed --lr 0.005 --local_layers 2 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.5 --method gat --rand_split --seed 123 --device 1 --runs 1  --kmeans 1  --num_codes 6 --epoch 1000 

python ID_MLP.py --dataset pubmed --lr 0.005 --num_layers 5 \
        --hidden_channels 256 --weight_decay 5e-4 --dropout 0.5 \
        --rand_split \
        --seed 123 --device 1 --runs 5 --num_id 6 --k 0 --epoch 1000

python main.py --dataset chameleon --lr 0.01 --local_layers 3 --hidden_channels 64 --weight_decay 0.001 --dropout 0.0 --method gat --device 4 --runs 1 --num_codes 32 --epoch 1000 --kmeans 1

python ID_MLP.py --dataset chameleon --lr 0.001 --num_layers 5 \
        --hidden_channels 512 --weight_decay 0.001 --dropout 0.5 \
        --device 4 --runs 5 --num_id 9 --k 0 --epoch 1000

python main.py --dataset squirrel --lr 0.005 --local_layers 6 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.0 --method gat --device 5 --runs 1 --num_codes 32 --kmeans 1

python ID_MLP.py --dataset squirrel --lr 0.001 --num_layers 2 \
        --hidden_channels 128 --weight_decay 5e-4 --dropout 0.5 \
        --device 5 --runs 5 --num_id 18 --k 0 --epoch 1000

# GCN
python main.py --dataset cora --lr 0.001 --local_layers 5 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.0 --method gcn --rand_split --seed 123 --device 2 --runs 1 --num_codes 6 --epoch 1000 --kmeans 1

python ID_MLP.py --dataset cora --lr 0.001 --num_layers 2 \
        --hidden_channels 512 --weight_decay 5e-4 --dropout 0.5 \
        --rand_split \
        --seed 123 --device 2 --runs 5 --num_id 15 --k 0 --epoch 1000

python main.py --dataset citeseer --lr 0.005 --local_layers 5 --hidden_channels 64 --weight_decay 0.01 --dropout 0.0 --method gcn --rand_split --seed 123 --device 3 --runs 1 --num_codes 16 --epoch 1000 --kmeans 1

python ID_MLP.py --dataset citeseer --lr 0.001 --num_layers 5 \
        --hidden_channels 256 --weight_decay 0.01 --dropout 0.5 \
        --rand_split \
        --seed 123 --device 3 --runs 5 --num_id 15 --k 0 --epoch 1000

python main.py --dataset pubmed --lr 0.01 --local_layers 3 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.5 --method gcn --rand_split --seed 123 --device 1 --runs 1  --kmeans 1  --num_codes 6 --epoch 1000 

python ID_MLP.py --dataset pubmed --lr 0.005 --num_layers 5 \
        --hidden_channels 256 --weight_decay 5e-4 --dropout 0.5 \
        --rand_split \
        --seed 123 --device 1 --runs 5 --num_id 9 --k 0 --epoch 1000

python main.py --dataset chameleon --lr 0.005 --local_layers 3 --hidden_channels 256 --weight_decay 0.001 --dropout 0.0 --method gcn --device 4 --runs 1 --num_codes 32 --epoch 1000 --kmeans 1

python ID_MLP.py --dataset chameleon --lr 0.001 --num_layers 2 \
        --hidden_channels 512 --weight_decay 0.001 --dropout 0.5 \
        --device 4 --runs 5 --num_id 9 --k 0 --epoch 1000

python main.py --dataset squirrel --lr 0.005 --local_layers 6 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 --method gcn --device 5 --runs 1 --num_codes 8 --kmeans 1

python ID_MLP.py --dataset squirrel --lr 0.001 --num_layers 2 \
        --hidden_channels 128 --weight_decay 5e-4 --dropout 0.5 \
        --device 5 --runs 5 --num_id 18 --k 0 --epoch 1000
