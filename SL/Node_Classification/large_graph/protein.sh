# ogbn-proteins
python -u protein_pre.py --gnum_layers 4 --gdropout 0.5 --device 0 --epoch 1000 --kmeans 1 --num_codes 4
python protein_ID_MLP.py --hidden_channels 512 --lr 0.001 --num_layers 5 --num_id 12 --norm_type batch --device 1
