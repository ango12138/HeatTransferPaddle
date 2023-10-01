# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/10/1 15:49
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：run_train.py
@File ：run_train.py
"""

import os, sys, random, math
import yaml
import numpy as np
import paddle
from process_data import HeatDataset, HeatDataLoader
from neural_model import BasicModule


with open(os.path.join('./all_config.yml')) as f:
    config = yaml.full_load(f)

general_config = config['general_config']
network_config = config['FNO_model']

data_file = os.path.join('./data', 'dim_pro8_single_try.mat')
train_dataset = HeatDataset(data_file, shuffle=False)
valid_dataset = HeatDataset(data_file, shuffle=False)
train_loader = HeatDataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)
valid_loader = HeatDataLoader(valid_dataset, batch_size=128, shuffle=False, drop_last=False)

Module = BasicModule('FNO', general_config, network_config=network_config)
Module.train(train_loader, valid_loader)
