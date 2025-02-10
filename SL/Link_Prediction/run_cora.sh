for kmeans in 0 1
do
for codebook in 16 32
do
for epoch in 150
do
for dropout1 in 0.1
do
for layer in 3 5 10
do

python ID_pretrain.py   --xdp 0.5 --tdp $dropout1 --gnndp $dropout1 --gnnedp $dropout1 \
--predp $dropout1 --preedp $dropout1 --gnnlr 0.004 \
--prelr 0.002  --batch_size 1152  --ln --lnnn --predictor cn1 --dataset Cora  --epochs $epoch \
--runs 1 --model gcn --hiddim 256 --mplayers $layer  --testbs 8192  \
--maskinput  --jk --codebook $codebook --kmeans $kmeans --tailact  --device 2

for dropout in 0.1 0.4
do
  for lr in 0.01 0.001
  do
    for batch_size in 1152
    do
      for hiddim in 512
      do

        python ID_MLP.py  --preedp $dropout --predp $dropout \
          --prelr 0.01  --batch_size $batch_size \
          --lnnn --predictor cn1 --dataset Cora  --epochs 1000 --runs 2 \
          --hiddim $hiddim --testbs 8192 --maskinput --num_id $((layer * 3)) --tailact  --device 2
          
      done
    done
  done
done


python ID_pretrain.py   --xdp 0.5 --tdp $dropout1 --gnndp $dropout1 --gnnedp $dropout1 \
--predp $dropout1 --preedp $dropout1 --gnnlr 0.004 \
--prelr 0.002  --batch_size 1152  --ln --lnnn --predictor cn1 --dataset Cora  --epochs $epoch \
--runs 1 --model gcn --hiddim 256 --mplayers $layer  --testbs 8192  \
--maskinput  --codebook $codebook --kmeans $kmeans --tailact  --device 2


for dropout in 0.1 0.4
do
  for lr in 0.01 0.001
  do
    for batch_size in 1152
    do
      for hiddim in 512
      do

        python ID_MLP.py  --preedp $dropout --predp $dropout \
          --prelr 0.01  --batch_size $batch_size \
          --lnnn --predictor cn1 --dataset Cora  --epochs 1000 --runs 2 \
          --hiddim $hiddim --testbs 8192 --maskinput --num_id $((layer * 3)) --tailact  --device 2
          
      done
    done
  done
done


done
done
done
done
done