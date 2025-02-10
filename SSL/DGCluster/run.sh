datasets="cora citeseer pubmed computers photo coauthorphysics"
lams="0.8"
seeds="0 1 2 3 4 5 6 7 8 9"

for dataset in $datasets; do
  for lam in $lams; do
    for seed in $seeds; do
      python main.py --dataset $dataset --lam $lam --seed $seed --device cuda:0
    done
  done
done