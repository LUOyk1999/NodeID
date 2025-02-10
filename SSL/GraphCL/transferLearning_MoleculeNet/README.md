## Dependencies & Dataset

Please refer to https://github.com/snap-stanford/pretrain-gnns#installation for environment setup and https://github.com/snap-stanford/pretrain-gnns#dataset-download to download dataset. Download the dataset, place it in `./chem/` and unzip it.

If you cannot manage to install the old torch-geometric version, one alternative way is to use the new one (maybe ==1.6.0) and make some modifications based on this issue https://github.com/snap-stanford/pretrain-gnns/issues/14.

## Training & Evaluation
### Pre-training: ###
```
cd ./chem
python pretrain_graphcl.py --aug1 random --aug2 none
```

### Node ID prediction: ###
```
cd ./chem
./run.sh
```