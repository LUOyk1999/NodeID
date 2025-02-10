# collab
python ID_pretrain.py   --xdp 0.25 --tdp 0.05 --gnnedp 0.25 --preedp 0.0 --predp 0.3 --gnndp 0.1  --gnnlr 0.001 --prelr 0.001  --batch_size 65536  --ln --lnnn --predictor cn1 --dataset collab  --epochs 150 --runs 1 --model gcn --hiddim 256 --mplayers 5  --testbs 131072  --maskinput --use_valedges_as_input --jk --device 4 --tailact

python ID_MLP.py  --preedp 0.0 --predp 0.0   --prelr 0.001  --batch_size 40000   --ln --lnnn --predictor cn1 --dataset collab  --epochs 200 --runs 1 --model gcn --hiddim 256   --testbs 131072  --maskinput --use_valedges_as_input   --device 4 --num_id 15  --tailact

# ppa
python ID_pretrain.py  --xdp 0.0 --tdp 0.0 --gnnedp 0.1 --preedp 0.0 --predp 0.1 --gnndp 0.0 --gnnlr 0.001 --prelr 0.001  --batch_size 16384  --ln --lnnn --predictor cn1 --dataset ppa   --epochs 60 --runs 1 --model gcn --hiddim 64 --mplayers 5 --maskinput  --tailact  --testbs 65536 --device 7 --res

python ID_MLP.py  --preedp 0.0 --predp 0.1 \
 --prelr 0.005  --batch_size 56384  --ln --lnnn --predictor cn1 --dataset ppa  \
  --epochs 100 --runs 1 --model gcn --hiddim 256 --maskinput  --tailact \
   --testbs 65536 --device 4 --num_id 15

# Cora
python ID_pretrain.py   --xdp 0.5 --tdp 0.1 --gnndp 0.1 --gnnedp 0.1 \
--predp 0.1 --preedp 0.1 --gnnlr 0.004 \
--prelr 0.002  --batch_size 1152  --ln --lnnn --predictor cn1 --dataset Cora  --epochs 150 \
--runs 1 --model gcn --hiddim 256 --mplayers 10  --testbs 8192  \
--maskinput  --jk --codebook 32 --kmeans 1 --tailact --device 3

python ID_MLP.py  --preedp 0.4 --predp 0.4 \
          --prelr 0.01  --batch_size 1152 \
          --lnnn --predictor cn1 --dataset Cora  --epochs 1000 --runs 2 \
          --hiddim 512 --testbs 8192 --maskinput --num_id 30 --tailact  --device 3

# Citeseer
python ID_pretrain.py   --xdp 0.4 --tdp 0.3 --gnndp 0.3 --gnnedp 0.3 \
--predp 0.3 --preedp 0.3 --gnnlr 0.01 \
--prelr 0.01  --batch_size 384  --ln --lnnn --predictor cn1 --dataset Citeseer  --epochs 10 \
--runs 1 --model puregcn --hiddim 256 --mplayers 10  --testbs 4096  \
--maskinput  --codebook 8 --kmeans 1 --tailact --device 0

python ID_MLP.py  --preedp 0.1 --predp 0.1 \
          --prelr 0.01 --batch_size 384 \
          --lnnn --predictor cn1 --dataset Citeseer  --epochs 1000 --runs 2 \
          --hiddim 512 --testbs 8192 --maskinput --num_id 30 --tailact --device 0

# Pubmed
python ID_pretrain.py   --xdp 0.5  --tdp 0.0 --gnndp 0.1 --gnnedp 0.0 \
--predp 0.0 --preedp 0.0 --gnnlr 0.01 \
--prelr 0.002 --batch_size 2048  --ln --lnnn --predictor cn1 --dataset Pubmed  --epochs 100 \
--runs 1 --model gcn --hiddim 256 --mplayers 10  --testbs 8192  \
--maskinput  --jk --use_xlin --codebook 8 --kmeans 1 --tailact --device 0

python ID_MLP.py  --preedp 0.3 --predp 0.3 \
  --prelr 0.001  --batch_size 4000 \
  --lnnn --predictor cn1 --dataset Pubmed  --epochs 1000 --runs 2 \
  --hiddim 512 --testbs 8192 --maskinput --num_id 30 --tailact  --device 0
