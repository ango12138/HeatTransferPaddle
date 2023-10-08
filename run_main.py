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
from process_data import HeatDataset, HeatDataLoader
from neural_model import BasicModule

# 读入原始配置文件
with open(os.path.join('DON.yaml'), encoding='utf-8') as f:
    config = yaml.full_load(f)

basic_config = config['basic_config']

# 数据配置
training_size = basic_config['training_size']
batch_size = basic_config['batch_size']
data_name = basic_config['data_name']

data_file = os.path.join('./data', data_name)
# data_file = os.path.join('./data', 'dim_pro8_single_try.mat')  # 小规模测试用
train_dataset = HeatDataset(data_file, training_size=training_size, mode=0, shuffle=False)
valid_dataset = HeatDataset(data_file, training_size=training_size, mode=1, shuffle=False)
train_loader = HeatDataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
valid_loader = HeatDataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# 网络参数配置
name = 'DON'
network_config = config[name + '_model']
Module = BasicModule(name=name, config=basic_config, network_config=network_config)

# 训练过程
Module.train(train_loader, valid_loader)

# 推理测试
test_dataset = HeatDataset(data_file, training_size=training_size, mode=2, shuffle=False)
test_loader = HeatDataLoader(test_dataset, batch_size=batch_size*2, shuffle=False, drop_last=False)

Module.infer(train_loader, 'train')
Module.infer(valid_loader, 'valid')
Module.infer(test_loader, 'test')