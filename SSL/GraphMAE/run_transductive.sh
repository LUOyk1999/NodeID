# Node classification results in unsupervised representation learning

python main_transductive.py \
	--device 0 \
	--dataset cora \
	--num_codes 32 \
	--num_layers 2 \
	--num_heads 4 \
	--num_hidden 1024 \
	--use_cfg 

python main_transductive.py \
	--device 0 \
	--dataset citeseer \
	--num_codes 8 \
	--num_layers 2 \
	--num_heads 2 \
	--num_hidden 256 \
	--use_cfg 

python main_transductive.py \
	--device 2 \
	--dataset pubmed \
	--num_codes 16 \
	--num_layers 2 \
	--num_heads 1 \
	--num_hidden 128 \
	--use_cfg 

# for num_hidden in 1024 512 256 128
# do
# for num_heads in 4 2 1
# do
# for num_codes in 16 32 8
# do
#     for num_layers in 2 3 4
#     do
# python -u main_transductive.py \
# 	--device $2 \
# 	--dataset $1 \
# 	--num_codes $num_codes \
# 	--num_layers $num_layers \
# 	--num_heads $num_heads \
# 	--num_hidden $num_hidden \
# 	--use_cfg 
# done
# done
# done
# done