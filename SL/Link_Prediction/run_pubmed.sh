for kmeans in 1 0
do
for codebook in 8 16 32
do
for epoch in 100
do
for dropout1 in 0.5
do
for layer in 5 10 15 20
do

python ID_pretrain.py   --xdp $dropout1 --tdp 0.0 --gnndp 0.1 --gnnedp 0.0 \
--predp 0.0 --preedp 0.0 --gnnlr 0.01 \
--prelr 0.002 --batch_size 2048  --ln --lnnn --predictor cn1 --dataset Pubmed  --epochs $epoch \
--runs 1 --model gcn --hiddim 256 --mplayers $layer  --testbs 8192  \
--maskinput  --jk --use_xlin --codebook $codebook --kmeans $kmeans --tailact --device 6

for dropout in 0.3 0.5
do
  for lr in 0.001
  do
    for batch_size in 6000
    do
      for hiddim in 512
      do
        python ID_MLP.py  --preedp $dropout --predp $dropout \
          --prelr $lr  --batch_size $batch_size \
          --lnnn --predictor cn1 --dataset Pubmed  --epochs 1500 --runs 2 \
          --hiddim $hiddim --testbs 8192 --maskinput --num_id $((layer * 3)) --tailact  --device 6
          
      done
    done
  done
done


python ID_pretrain.py   --xdp $dropout1 --tdp 0.0 --gnndp 0.1 --gnnedp 0.0 \
--predp 0.0 --preedp 0.0 --gnnlr 0.01 \
--prelr 0.002 --batch_size 2048  --ln --lnnn --predictor cn1 --dataset Pubmed  --epochs $epoch \
--runs 1 --model gcn --hiddim 256 --mplayers $layer  --testbs 8192  \
--maskinput  --jk --codebook $codebook --kmeans $kmeans --tailact --device 6

for dropout in 0.3 0.5
do
  for lr in 0.001
  do
    for batch_size in 6000
    do
      for hiddim in 512
      do
    
        python ID_MLP.py  --preedp $dropout --predp $dropout \
          --prelr $lr  --batch_size $batch_size \
          --lnnn --predictor cn1 --dataset Pubmed  --epochs 1500 --runs 2 \
          --hiddim $hiddim --testbs 8192 --maskinput --num_id $((layer * 3)) --tailact  --device 6
          
      done
    done
  done
done

done
done
done
done
done

python ID_pretrain.py   --xdp 0.5  --tdp 0.0 --gnndp 0.1 --gnnedp 0.0 \
--predp 0.0 --preedp 0.0 --gnnlr 0.01 \
--prelr 0.002 --batch_size 2048  --ln --lnnn --predictor cn1 --dataset Pubmed  --epochs 100 \
--runs 1 --model gcn --hiddim 256 --mplayers 10  --testbs 8192  \
--maskinput  --jk --use_xlin --codebook 8 --kmeans 1 --tailact --device 0

python ID_MLP.py  --preedp 0.3 --predp 0.3 \
  --prelr 0.001  --batch_size 4000 \
  --lnnn --predictor cn1 --dataset Pubmed  --epochs 1000 --runs 2 \
  --hiddim 512 --testbs 8192 --maskinput --num_id 30 --tailact  --device 0