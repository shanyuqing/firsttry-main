#! /bin/bash
# mlp在不同hidden的简单对比
python ../train.py --cfg_file 'cfgs/MLP_miner/hid_64.yaml' 
python ../train.py --cfg_file 'cfgs/MLP_miner/hid_256.yaml' 