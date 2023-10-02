
# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/9/30 9:29
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：deepONet.py
@File ：deepONet.py
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from utilize import activation_dict, params_initial
from typing import Any, List, Tuple, Union

class MLP(nn.Layer):
    def __init__(self, planes: list or tuple, activation="gelu", last_activation=False):
        # =============================================================================
        #     Inspired by M. Raissi a, P. Perdikaris b,∗, G.E. Karniadakis.
        #     "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems
        #     involving nonlinear partial differential equations".
        #     Journal of Computational Physics.
        # =============================================================================
        super(MLP, self).__init__()
        self.planes = planes
        self.active = activation_dict[activation]

        self.layers = nn.LayerList()
        for i in range(len(self.planes) - 2):
            self.layers.append(nn.Linear(self.planes[i], self.planes[i + 1]))
            self.layers.append(self.active)
        self.layers.append(nn.Linear(self.planes[-2], self.planes[-1]))

        if last_activation:
            self.layers.append(self.active)
        self.layers = nn.Sequential(*self.layers)  # *的作用是解包

        # self.reset_parameters()

    def reset_parameters(self):
        """
        weight initialize
        """
        for m in self.sublayers():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight, gain=1)
                w = params_initial('xavier_normal', shape=m.weight.shape)
                m.weight.set_value(w)
                b = params_initial('constant', shape=m.bias.shape)
                m.bias.set_value(b)

    def forward(self, in_var):
        """
        forward compute
        :param in_var: (batch_size, ..., input_dim)
        """
        out_var = self.layers(in_var)
        return out_var


class FcnMulti(nn.Layer):
    def __init__(self, in_dim, out_dim, planes: list, steps=1, activation="gelu"):
        # =============================================================================
        #     Inspired by Haghighat Ehsan, et all.
        #     "A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics"
        #     Computer Methods in Applied Mechanics and Engineering.
        # =============================================================================
        super(FcnMulti, self).__init__()
        self.planes = [steps * in_dim + 2,] + planes + [out_dim]
        self.active = activation_dict[activation]

        self.layers = nn.LayerList()
        for j in range(self.planes[-1]):
            layer = []
            for i in range(len(self.planes) - 2):
                layer.append(nn.Linear(self.planes[i], self.planes[i + 1]))
                layer.append(self.active)
            layer.append(nn.Linear(self.planes[-2], 1))
            self.layers.append(nn.Sequential(*layer))
        self.reset_parameters()

    def reset_parameters(self):
        """
        weight initialize
        """
        for m in self.sublayers():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight, gain=1)
                w = params_initial('xavier_normal', shape=m.weight.shape)
                m.weight.set_value(w)
                b = params_initial('constant', shape=m.bias.shape)
                m.bias.set_value(b)

    def forward(self, x, grid):
        """
        forward compute
        :param in_var: (batch_size, ..., input_dim)
        """

        if len(x.shape) != len(grid.shape):
            repeat_times = paddle.to_tensor([1]+grid.shape[1:-1]+[1], dtype='int32')
            x = paddle.tile(x[:, None, None, :], repeat_times=repeat_times)

        in_var = paddle.concat((x, grid), axis=-1)

        y = []
        for i in range(self.planes[-1]):
            y.append(self.layers[i](in_var))
        return paddle.concat(y, axis=-1)

class DeepONetMulti(nn.Layer):
    # =============================================================================
    #     Inspired by L. Lu, J. Pengzhan, G.E. Karniadakis.
    #     "DeepONet: Learning nonlinear operators for identifying differential equations based on
    #     the universal approximation theorem of operators".
    #     arXiv:1910.03193v3 [cs.LG] 15 Apr 2020.
    # =============================================================================
    def __init__(self, in_dim: int, out_dim: int, operator_dims: list,
                 planes_branch: list, planes_trunk: list, activation='gelu'):
        """
        :param in_dim: int, the coordinates dim for trunk net
        :param operator_dims: list，the operate dims list for each branch net
        :param out_dim: int, the predicted variable dims
        :param planes_branch: list, the hidden layers dims for branch net
        :param planes_trunk: list, the hidden layers dims for trunk net
        :param operator_dims: list，the operate dims list for each branch net
        :param activation: activation function
        """
        super(DeepONetMulti, self).__init__()

        self.branches = nn.LayerList() # 分支网络
        self.trunks = nn.LayerList() # 主干网络
        for dim in operator_dims:
            self.branches.append(MLP([dim] + planes_branch, activation=activation))# FcnSingle是从basic_layers里导入的
        for _ in range(out_dim):
            self.trunks.append(MLP([in_dim] + planes_trunk, activation=activation))

        self.reset_parameters()

    def reset_parameters(self):
        """
        weight initialize
        """
        for m in self.sublayers():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight, gain=1)
                w = params_initial('xavier_normal', shape=m.weight.shape)
                m.weight.set_value(w)
                b = params_initial('constant', shape=m.bias.shape)
                m.bias.set_value(b)

    def forward(self, u_vars, y_var, size_set=False):
        """
        forward compute
        :param u_vars: tensor list[(batch_size, ..., operator_dims[0]), (batch_size, ..., operator_dims[1]), ...]
        :param y_var: (batch_size, ..., input_dim)
        :param size_set: bool, true for standard inputs, false for reduce points number in operator inputs
        """
        B = 1.

        if not isinstance(u_vars, list or tuple):
            u_vars = [u_vars,]

        for u_var, branch in zip(u_vars, self.branches):
            B *= branch(u_var)

        if not size_set:
            B_size = list(y_var.shape[1:-1])
            for i in range(len(B_size)):
                B = B.unsqueeze(1)
            B = paddle.tile(B, [1, ] + B_size + [1, ])

        out_var = []
        for trunk in self.trunks:
            T = trunk(y_var)
            out_var.append(paddle.sum(B * T, axis=-1)) # 用这种方式实现两个网络的乘积
        out_var = paddle.stack(out_var, axis=-1)
        return out_var


if __name__ == "__main__":
    us = [paddle.ones([10, 256 * 2]), paddle.ones([10, 1])]
    x = paddle.ones([10, 2])
    layer = DeepONetMulti(in_dim=2, out_dim=5, operator_dims=[256 * 2, 1],
                          planes_branch=[64] * 3, planes_trunk=[64] * 2)
    y = layer(us, x)
    print(y.shape)

    us = paddle.ones([10, 10])
    x = paddle.ones([10, 64, 64, 2])
    layer = DeepONetMulti(in_dim=2, out_dim=5, operator_dims=[10,],
                          planes_branch=[64] * 3, planes_trunk=[64] * 2)
    y = layer(us, x, size_set=False)
    print(y.shape)


    x = paddle.ones([10, 3])
    g = paddle.ones([10, 64, 64, 2])
    layer = FcnMulti(in_dim=3, out_dim=5, steps=1, planes=[64, 64])
    y = layer(x, g)
    print(y.shape)

