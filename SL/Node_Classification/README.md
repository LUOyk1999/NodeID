## Datasets

Chameleon and Squirrel: one can download the datasets from the google drive link below:
https://drive.google.com/drive/folders/1rr3kewCBUvIuVxA6MJ90wzQuF-NnCRtf?usp=drive_link (provided by Qitian Wu and Wentao Zhao and Chenxiao Yang and Hengrui Zhang and Fan Nie and Haitian Jiang and Yatao Bian and Junchi Yan, Simplifying and empowering transformers for large-graph representations. In Thirty-seventh Conference on Neural Information Processing Systems, 2023b.)

Download the geom-gcn folder, place it in `./data/` and unzip it.

For Chameleon and Squirrel, we use the [new splits](https://github.com/yandex-research/heterophilous-graphs/tree/main) that filter out the overlapped nodes.
Download `chameleon_filtered.npz`, put it into `./data/geom-gcn/chameleon/`.
Download `squirrel_filtered.npz`, put it into `./data/geom-gcn/squirrel/`.

## Note

```bash
sh run.sh

cd large_graph_exp
bash arxiv.sh
bash pokec.sh
bash product.sh
bash protein.sh
```