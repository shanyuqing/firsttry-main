#! /bin/bash
# mlp在不同activation的简单对比
python ../train.py --cfg_file 'cfgs/MLP_miner/act_exp/GELU.yaml' --extra_tag 'act'
python ../train.py --cfg_file 'cfgs/MLP_miner/act_exp/Sigmoid.yaml' --extra_tag 'act'
python ../train.py --cfg_file 'cfgs/MLP_miner/act_exp/tanh.yaml' --extra_tag 'act'
python ../train.py --cfg_file 'cfgs/MLP_miner/act_exp/ReLU.yaml' --extra_tag 'act'
python ../train.py --cfg_file 'cfgs/MLP_miner/act_exp/HardSwish.yaml' --extra_tag 'act'