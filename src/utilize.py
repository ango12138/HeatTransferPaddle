# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/9/30 9:31
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：utilize.py
@File ：utilize.py
"""
import os
import numpy as np
import paddle
import pickle
import paddle.nn as nn
from collections import defaultdict

activation_dict = \
    {'gelu': nn.GELU(), 'relu': nn.ReLU(),
     'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'softplus': nn.Softplus(),
     'leakyrelu': nn.LeakyReLU(), 'silu': nn.Silu(), 'elu': nn.ELU(),
     None: nn.GELU()}

lossfunc_dict = \
    {'mse': nn.MSELoss(), 'l2loss': nn.MSELoss(),
     'l1loss': nn.L1Loss(), 'mae': nn.L1Loss(),
     'smooth_l1loss': nn.SmoothL1Loss(), 'huber_loss': nn.SmoothL1Loss(), 'huber': nn.SmoothL1Loss(),
     }


def default(value, d):
    """
        helper taken from https://github.com/lucidrains/linear-attention-transformer
    """
    return d if value is None else value


def params_initial(initialization, shape, scale=1.0, gain=1.0):
    if initialization == 'constant':
        Weight = gain * np.ones(shape).astype('float32')
    elif initialization == 'normal':
        Weight = gain * np.random.normal(loc=0., scale=scale, size=shape).astype('float32')
    elif initialization == 'xavier_Glorot_normal':
        in_dim, out_dim = calculate_fan_in_and_fan_out(shape)
        Weight = gain * np.random.normal(loc=0., scale=scale, size=shape) / np.sqrt(in_dim).astype('float32')
    elif initialization == 'xavier_normal':
        in_dim, out_dim = calculate_fan_in_and_fan_out(shape)
        std = np.sqrt(2. / (in_dim + out_dim))
        Weight = gain * np.random.normal(loc=0., scale=std, size=shape).astype('float32')
    elif initialization == 'uniform':
        in_dim, out_dim = calculate_fan_in_and_fan_out(shape)
        a = np.sqrt(1. / in_dim)
        Weight = gain * np.random.uniform(low=-a, high=a, size=shape).astype('float32')
    elif initialization == 'xavier_uniform':
        in_dim, out_dim = calculate_fan_in_and_fan_out(shape)
        a = np.sqrt(6. / (in_dim + out_dim))
        Weight = gain * np.random.uniform(low=-a, high=a, size=shape).astype('float32')
    else:
        print("initialization error!")
        exit(1)
    return Weight

def calculate_fan_in_and_fan_out(shape):
    # dimensions = tensor.dim()
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = shape[1]
    num_output_fmaps = shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def makeDirs(directoryList):

    if not isinstance(directoryList, list or tuple):
        directoryList = [directoryList,]

    for directory in directoryList:
        if not os.path.exists(directory):
            os.makedirs(directory)


class LogHistory(object):

    def __init__(self, log_names: list or tuple, **kwargs):

        self.epoch_list = []
        self.time_train = []
        self.time_valid = []
        self.loss_train = []
        self.loss_valid = []
        self.metric_train = []
        self.metric_valid = []
        self.log_names = ['epoch_list', 'time_train', 'time_valid',
                          'loss_train', 'loss_valid', 'metric_train', 'metric_valid']

        assert log_names is not list or tuple or None, "log_names must be list or tuple"
        log_names = [] if log_names is None else log_names

        temp_names = defaultdict(lambda: None, **kwargs)
        self.log_names = self.log_names + list(log_names) + list(temp_names.keys())
        self._set_default()

    def _set_default(self):
        all_attr = self.log_names
        for key in all_attr:
            setattr(self, key, [])

    def append(self, epoch, **kwargs):
        self.epoch_list.append(epoch)

        for key, value in kwargs.items():
            if key in self.log_names:
                self.__dict__[key].append(value)
            else:
                assert key in self.log_names, "{} must be in log_names!".format(str(key))

    def save(self, file):

        loghistory = dict((key, self.__dict__[key]) for key in self.log_names)
        loghistory['log_names'] = self.log_names
        with open(file, 'wb') as f:
            pickle.dump(loghistory, f)

    def load(self, file):
        isExist = os.path.exists(file)
        if isExist:
            with open(file, 'rb') as file:
                loghistory = pickle.loads(file.read())
            for key in loghistory.keys():
                self.__dict__[key] = loghistory[key]
        else:
            raise ValueError("The pkl file is not exist, CHECK PLEASE!")

if __name__ == '__main__':

    a = LogHistory(log_names=['metric_train', 'metric_valid'], metric_fields_list=['a'])

    a.append(1, 1, 1, metric_train=[1, 2, 3], metric_valid=[4, 5, 6])

    a=0