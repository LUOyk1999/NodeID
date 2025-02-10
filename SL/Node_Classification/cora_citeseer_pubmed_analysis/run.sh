# Cora
python main.py --dataset cora --lr 0.01 --num_layers 4 \
        --hidden_channels 128 --weight_decay 5e-4 --dropout 0.0 \
        --method gcn \
        --rand_split --no_feat_norm \
        --seed 123 --device 2 --runs 1 --num_codes 6 --epoch 1000 --kmeans 1

python ID_MLP.py --dataset cora --lr 0.001 --num_layers 5 \
                --hidden_channels 256 --weight_decay 5e-4 --dropout 0.5 \
                --rand_split --no_feat_norm \
                --seed 123 --device 2 --runs 5 --num_id 12 --k 0 --epoch 1000

# Citeseer
python main.py --dataset citeseer --lr 0.01 --num_layers 2 \
        --hidden_channels 128 --weight_decay 0.01 --dropout 0.0 \
        --method gcn \
        --rand_split --no_feat_norm \
        --seed 123 --device 4 --runs 1 --num_codes 8 --epoch 1000 --kmeans 1

python ID_MLP.py --dataset citeseer --lr 0.001 --num_layers 5 \
                --hidden_channels 256 --weight_decay 0.01 --dropout 0.5 \
                --rand_split --no_feat_norm \
                --seed 123 --device 4 --runs 5 --num_id 6 --k 0 --epoch 1000

# Pubmed
python main.py --dataset pubmed --lr 0.005 --num_layers 2 \
        --hidden_channels 256 --weight_decay 5e-4 --dropout 0.5 \
        --method gcn \
        --rand_split --no_feat_norm \
        --seed 123 --device 3 --runs 1  --kmeans 1  --num_codes 16 --epoch 1000 

python ID_MLP.py --dataset pubmed --lr 0.005 --num_layers 5 \
            --hidden_channels 256 --weight_decay 5e-4 --dropout 0.5 \
            --rand_split --no_feat_norm \
            --seed 123 --device 3 --runs 5 --num_id 6 --k 0 --epoch 1000

