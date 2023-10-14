# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/10/1 15:49
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：run_infer.py
@File ：run_infer.py
"""

import os, sys, random, math
import yaml
from process_data import HeatDataset, HeatDataLoader
from neural_model import BasicModule

# 读入原始配置文件
name = 'TNO'
with open(os.path.join('config', name + '.yaml'), encoding='utf-8') as f:
    config = yaml.full_load(f)
basic_config = config['basic_config']

# 数据配置
training_size = basic_config['training_size']
batch_size = basic_config['batch_size']
data_name = basic_config['data_name']

data_file = os.path.join('./data', data_name)

# 网络参数配置
load_path = os.path.join('work', name, '2023-10-02-00-52')
Module = BasicModule(name=name, config=None, network_config=None, load_path=load_path)

# 推理测试
test_dataset = HeatDataset(data_file, training_size=training_size, mode=2, shuffle=False)
test_loader = HeatDataLoader(test_dataset, batch_size=batch_size*2, shuffle=False, drop_last=False)

Module.infer(test_loader, 'test')
