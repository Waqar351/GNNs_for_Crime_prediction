EvolveGCN
=====

This repository contains the code for [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/abs/1902.10191), published in AAAI 2020.

With some changes for the application in São Paulo city crime data.

## Data
 
For the downloaded and processed São Paulo data set, please place the processed data in the 'data/processed/sp' folder. After that just go to the EvolveGCN directory and run the following script:

```sh
python3 egcn_dataloader.py
```
After that the model can be properly used.

## Requirements
  All the requirements are listed in pyproject.toml file.

## Usage of the model

Set --config_file with a yaml configuration file to run the experiments. For example:

```sh
python3 run_sp_exp.py --config_file ./experiments/parameters_example.yaml
```

Most of the parameters in the yaml configuration file are self-explanatory. For hyperparameters tuning, it is possible to set a certain parameter to 'None' and then set a min and max value. Then, each run will pick a random value within the boundaries (for example: 'learning_rate', 'learning_rate_min' and 'learning_rate_max').

Setting 'use_logfile' to True in the configuration yaml will output a file, in the 'log' directory, containing information about the experiment and validation metrics for the various epochs. The file could be manually analyzed, alternatively 'log_analyzer.py' can be used to automatically parse a log file and to retrieve the evaluation metrics at the best validation epoch (Not used for the purposes of our paper, but recommended by the authors of EvolveGCN paper). For example:
```sh
python log_analyzer.py log/filename.log
```

Besides the metrics results, to reproduce the colormap you can go to the folder "color_map", make sure to put in this folder the file "MAP_SOFT_PREDICTIONS.txt" generated after the model finished running and the file "street_nodes_dataframe.pickle" of the processed data. There will be a jupyter notebook to create the map with the help of this two data files, just run all the cells and the html archive will be downloaded in the same folder.

## Reference

[1] Aldo Pareja, Giacomo Domeniconi, Jie Chen, Tengfei Ma, Toyotaro Suzumura, Hiroki Kanezashi, Tim Kaler, Tao B. Schardl, and Charles E. Leiserson. [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/abs/1902.10191). AAAI 2020.

## BibTeX entry

Please cite the paper if you use this code in your work:

```
@INPROCEEDINGS{egcn,
  AUTHOR = {Aldo Pareja and Giacomo Domeniconi and Jie Chen and Tengfei Ma and Toyotaro Suzumura and Hiroki Kanezashi and Tim Kaler and Tao B. Schardl and Charles E. Leiserson},
  TITLE = {{EvolveGCN}: Evolving Graph Convolutional Networks for Dynamic Graphs},
  BOOKTITLE = {Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence},
  YEAR = {2020},
}
```
