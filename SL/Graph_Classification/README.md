## Python environment setup with Conda

```bash
conda create -n graphgps python=3.10
conda activate graphgps

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

conda install openbabel fsspec rdkit -c conda-forge

pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb

conda clean --all
```


## Running Training
```bash
conda activate graphgps
python main.py --cfg configs/LRGB-tuned/peptides-struct-GCN.yaml wandb.use False
python ID_MLP_s.py
python main.py --cfg configs/LRGB-tuned/peptides-func-GCN.yaml wandb.use False
python ID_MLP_f.py
```

