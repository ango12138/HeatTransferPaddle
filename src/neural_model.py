# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/9/30 15:09
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：run_train.py
@File ：run_train.py
"""

import os, sys, random, math
import datetime, time
from process_data import HeatDataset
from utilize import makeDirs, activation_dict, lossfunc_dict

import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optim

import matplotlib.pyplot as plt
from visual_data import MatplotlibVision
from utilize import LogHistory


class BasicModule(object):

    def __init__(self, name, config, network_config):

        self.name = name
        self.config = config
        self.network_config = network_config

        self._set_config()
        self._set_path()
        self._set_device()
        self._set_network()
        self._set_optim()
        self._set_logger()
        self._set_visual()

        self.characteristic = Characteristic()
        self.fields_metric = PhysicsLpLoss(p=2, samples_reduction=False, channel_reduction=False)  # 使用二阶范数
        self.target_metric = PhysicsLpLoss(p=1, samples_reduction=False, channel_reduction=False)  # 使用一阶范数

    def train(self, train_loader, valid_loader):

        for epoch in range(1, self.total_epoch + 1):

            sta_time = time.time()
            self.train_epoch(train_loader)
            train_epoch_time = time.time() - sta_time
            train_metric, train_loss = self.valid_epoch(train_loader, 'train')
            valid_train_time = time.time() - train_epoch_time - sta_time
            valid_metric, valid_loss = self.valid_epoch(valid_loader, 'valid')
            valid_valid_time = time.time() - valid_train_time - train_epoch_time - sta_time

            self.loghistory.append(epoch_list=epoch,
                                   time_train=train_epoch_time,
                                   time_valid=valid_train_time + valid_valid_time,
                                   loss_fields_train=train_loss['fields'], loss_flieds_valid=valid_loss['fields'],
                                   loss_target_train=train_loss['target'], loss_target_valid=valid_loss['target'],
                                   metric_fields_train=train_metric['fields'],
                                   metric_fields_valid=valid_metric['fields'],
                                   metric_target_train=train_metric['target'],
                                   metric_target_valid=valid_metric['target'])

            print('epoch: {:6d}, learning_rate: {:.3e}, '
                  'train_cost: {:.2f}, valid_cost: {:.2f}'
                  'train_epoch_loss: {:.3e}, valid_epoch_loss: {:.3e}'.
                  format(epoch, self.optimizer.get_lr(),
                         self.loghistory.time_train[-1], self.loghistory.time_valid[-1],
                         self.loghistory.loss_train[-1].mean(), self.loghistory.loss_train[-1].mean()
                         ))

            # print(os.environ['CUDA_VISIBLE_DEVICES'])
            star_time = time.time()

            if epoch > 0 and epoch % 5 == 0:
                fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
                self.visual.plot_loss(fig, axs, label='train',
                                      y=np.array(self.loghistory.loss_train).mean(axis=-1),
                                      std=np.array(self.loghistory.loss_train).std(axis=-1))
                self.visual.plot_loss(fig, axs, label='valid',
                                      y=np.array(self.loghistory.loss_valid).mean(axis=-1),
                                      std=np.array(self.loghistory.loss_valid).std(axis=-1))
                fig.suptitle('training process')
                fig.savefig(self.train_path, 'training_process.jpg', dpi=300)
                plt.close(fig)

            if epoch > 0 and epoch % 100 == 0:
                paddle.save({
                    'epoch': epoch,
                    'config': self.config,
                    'network_config': self.network_config,
                    'loghistory': self.loghistory,
                    'net_model': self.net_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()},
                    os.path.join(self.work_path, 'last_model.pdparams'))

    def valid(self):
        pass

    def infer(self):
        pass

    def train_epoch(self, train_loader):

        sta_time = time.time()
        self.net_model.train()
        for data in train_loader:
            design, coords, fields, _ = data
            self.optimizer.clear_grad()
            fields_ = self.net_model(design, coords)
            loss = self.loss_func(fields_, fields)
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()

    def valid_epoch(self, data_loader, data_name):

        log_metric = {'target': [], 'fields': []}
        log_loss = {'target': [], 'fields': []}

        sta_time = time.time()
        self.net_model.eval()
        with paddle.no_grad():
            for data in data_loader:
                design, coords, fields, _ = data

                fields_ = self.net_model(design, coords)

                design = data_loader.design_back(design)
                coords = data_loader.coords_back(coords)
                fields = data_loader.fields_back(fields)
                fields_ = data_loader.fields_back(fields_)

                target = self.characteristic(fields, coords, design)
                target_ = self.characteristic(fields_, coords, design)

                fields_loss = self.loss_func(fields_, fields).item()
                target_loss = self.loss_func(target_, target).item()

                fields_metric = self.fields_metric(fields_, fields).cpu().numpy()
                target_metric = self.target_metric(target_, target).cpu().numpy()

                log_metric['fields'].append(fields_metric)
                log_metric['target'].append(target_metric)

                log_loss['fields'].append(fields_loss)
                log_loss['target'].append(target_loss)

        log_metric['fields'] = np.concatenate(log_metric['fields'], axis=0)
        log_metric['target'] = np.concatenate(log_metric['target'], axis=0)

        log_loss['fields'] = np.array(log_loss['fields'])
        log_loss['target'] = np.array(log_loss['target'])

        return log_metric, log_loss

    def _set_config(self):
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])

    def _set_device(self):

        if paddle.device.is_compiled_with_cuda():
            device = paddle.device.set_device('gpu')
        else:
            device = paddle.device.set_device('cpu')
        self.device = device

    def _set_path(self):

        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
        self.work_path = os.path.join(self.root_path, 'work', self.name, formatted_datetime)
        self.train_path = os.path.join(self.work_path, 'train')
        self.valid_path = os.path.join(self.work_path, 'valid')
        self.infer_path = os.path.join(self.work_path, 'infer')

        makeDirs([self.work_path, self.train_path, self.valid_path, self.infer_path])

    def _set_network(self):
        if 'FNO' in self.name:
            from FNO_model import FNO2d
            self.net_model = FNO2d(**self.network_config)
        elif 'CNN' in self.name:
            from CNN_model import UNet2d
            self.net_model = UNet2d(**self.network_config)
        elif 'DON' in self.name:
            from DON_model import DeepONetMulti
            self.net_model = DeepONetMulti(**self.network_config)
        elif 'MLP' in self.name:
            from DON_model import FcnMulti
            self.net_model = FcnMulti(**self.network_config)
        elif 'TNO' in self.name:
            from TNO_model import FourierTransformer2D
            self.net_model = FourierTransformer2D(**self.network_config)
        self.net_model = self.net_model.to(self.device)

    def _set_optim(self):

        model_parameters = filter(lambda p: ~p.stop_gradient, self.net_model.parameters())
        params = sum([np.prod(p.shape) for p in model_parameters])
        print("Initialized {} with {} trainable params ".format(self.name, params))

        self.loss_func = lossfunc_dict[self.loss_name]
        self.optimizer = optim.Adam(parameters=self.net_model.parameters(),
                                    learning_rate=self.learning_rate,
                                    beta1=self.learning_beta[0], beta2=self.learning_beta[1],
                                    weight_decay=self.weight_decay)

        self.scheduler = optim.lr.MultiStepDecay(learning_rate=self.learning_rate,
                                                 milestones=self.learning_milestones,
                                                 gamma=self.learning_gamma)

    def _set_logger(self):

        self.loghistory = LogHistory(log_names=('fields', ''))

    def _set_visual(self):

        self.visual = MatplotlibVision(self.work_path, input_name=('x', 'y'), field_name=('P', 'T', 'U', 'V'))


class Characteristic(nn.Layer):

    def __init__(self):
        super(Characteristic, self).__init__()

    def get_parameters_of_nano(self, per):
        lamda_water = 0.597
        Cp_water = 4182.
        rho_water = 998.2
        miu_water = 9.93e-4

        lamda_al2o3 = 36.
        Cp_al2o3 = 773.
        rho_al2o3 = 3880.

        rho = per * rho_al2o3 + (1. - per) * rho_water
        Cp = ((1. - per) * rho_water * Cp_water + per * rho_al2o3 * Cp_al2o3) / rho
        miu = miu_water * (123. * per ** 2. + 7.3 * per + 1)
        DELTA = ((3. * per - 1.) * lamda_al2o3 + (2. - 3. * per) * lamda_water) ** 2
        DELTA = DELTA + 8. * lamda_al2o3 * lamda_water
        lamda = 0.25 * ((3 * per - 1) * lamda_al2o3 + (2 - 3 * per) * lamda_water + paddle.sqrt(DELTA))

        return lamda, Cp, rho, miu

    def cal_f(self, X, Y, P):
        F_inn = (P[:, 0, 1:] + P[:, 0, 0:-1]) / 2
        F_out = (P[:, -1, 1:] + P[:, -1, 0:-1]) / 2

        dy_inn = Y[:, 0, 1:] - Y[:, 0, 0:-1]
        dy_out = Y[:, -1, 1:] - Y[:, -1, 0:-1]

        D_P = paddle.sum(F_inn * dy_inn, axis=(1,)) / paddle.sum(dy_inn, axis=(1,)) \
              - paddle.sum(F_out * dy_out, axis=(1,)) / paddle.sum(dy_out, axis=(1,))

        return D_P

    def cal_tb(self, X, Y, T):
        F_T = T[:, :, :]

        dxx = X[:, :-1, :] - X[:, 1:, :]
        dxy = Y[:, :-1, :] - Y[:, 1:, :]
        dyx = X[:, :, 1:] - X[:, :, :-1]
        dyy = Y[:, :, 1:] - Y[:, :, :-1]

        ds1 = paddle.abs(dxx[:, :, :-1] * dyy[:, 1:] - dxy[:, :, :-1] * dyx[:, 1:]) / 2
        ds2 = paddle.abs(dxx[:, :, 1:] * dyy[:, :-1] - dxy[:, :, 1:] * dyx[:, :-1]) / 2
        ds = ds1 + ds2

        M_T = (F_T[:, 1:, 1:] + F_T[:, 1:, :-1] + F_T[:, :-1, 1:] + F_T[:, :-1, :-1]) / 4

        Tb = paddle.sum(ds * M_T, axis=(1, 2)) / paddle.sum(ds, axis=(1, 2))

        return Tb

    def cal_tw(self, X, Y, T):
        up_t = T[:, :, -1]
        down_t = T[:, :, 0]

        temp = paddle.sqrt((X[:, :-1, -1] - X[:, 1:, -1]) ** 2 + (Y[:, :-1, -1] - Y[:, 1:, -1]) ** 2)
        up_dl = paddle.zeros_like(X[:, :, 0])
        up_dl[:, 1:-1] = (temp[:, :-1] + temp[:, 1:]) / 2
        up_dl[:, 0] = temp[:, 0] / 2
        up_dl[:, -1] = temp[:, -1] / 2

        temp = paddle.sqrt((X[:, :-1, 0] - X[:, 1:, 0]) ** 2 + (Y[:, :-1, 0] - Y[:, 1:, 0]) ** 2)
        down_dl = paddle.zeros_like(X[:, :, 0])
        down_dl[:, 1:-1] = (temp[:, :-1] + temp[:, 1:]) / 2
        down_dl[:, 0] = temp[:, 0] / 2
        down_dl[:, -1] = temp[:, -1] / 2

        Tw = ((paddle.sum(up_t * up_dl, axis=1) + paddle.sum(down_t * down_dl, axis=1))
              / paddle.sum(up_dl + down_dl, axis=1))

        return Tw

    def forward(self, field, grid, design):
        D = float(2 * 200 * 1e-6)  # 水力直径
        L = float(3500 * 1e-6)  # 通道长度

        per = design[:, 3]  # Al2O3体积分数
        Re = design[:, 0]  # Reynaldo
        hflex = design[:, 2]  # 热流密度

        lamda, Cp, rho, miu = self.get_parameters_of_nano(per)

        I_ext = 121
        X = grid[:, I_ext:-I_ext, :, 0]
        Y = grid[:, I_ext:-I_ext, :, 1]
        P = field[:, I_ext:-I_ext, :, 0]
        T = field[:, I_ext:-I_ext, :, 1]

        Tw = self.cal_tw(X, Y, T)
        Tb = self.cal_tb(X, Y, T)
        h = hflex / paddle.abs(Tw - Tb)
        Nu = h * D / lamda

        vel = Re * miu / rho / D
        Dp = self.cal_f(X, Y, P)
        Fan = Dp * D / 2 / L / vel / vel / rho

        result = paddle.stack((Nu, Fan), axis=1)

        return result


# loss function with rel/abs Lp loss
class PhysicsLpLoss(object):
    def __init__(self, p=2, relative=True, samples_reduction=True, channel_reduction=False):
        super(PhysicsLpLoss, self).__init__()

        # Lp-norm type are positive
        assert p > 0, 'Lp-norm type should be positive!'

        self.p = p
        self.relative = relative
        self.channel_reduction = channel_reduction
        self.samples_reduction = samples_reduction

    def forward(self, x, y):

        if paddle.is_tensor(x):
            dif_norms = paddle.norm(x.reshape(x.shape[0], -1, x.shape[-1]) -
                                    y.reshape(x.shape[0], -1, x.shape[-1]), self.p, 1)

            all_norms = paddle.norm(y.reshape(x.shape[0], -1, x.shape[-1]), self.p, 1)

            if self.relative:
                res_norms = dif_norms / (all_norms + 1e-12)
            else:
                res_norms = dif_norms

            if self.samples_reduction:
                res_norms = paddle.mean(res_norms, axis=0)

            if self.channel_reduction:
                res_norms = paddle.mean(res_norms, axis=-1)

        else:
            dif_norms = np.linalg.norm(x.reshape(x.shape[0], -1, x.shape[-1]) -
                                       y.reshape(x.shape[0], -1, x.shape[-1]), self.p, 1)
            all_norms = np.linalg.norm(y.reshape(x.shape[0], -1, x.shape[-1]), self.p, 1)

            if self.relative:
                res_norms = dif_norms / (all_norms + 1e-12)
            else:
                res_norms = dif_norms

            if self.samples_reduction:
                res_norms = np.mean(res_norms, axis=0)

            if self.channel_reduction:
                res_norms = np.mean(res_norms, axis=-1)

        return res_norms

    def __call__(self, x, y):

        return self.forward(x, y)


if __name__ == "__main__":

    import yaml

    with open(os.path.join('../all_config.yml')) as f:
        config = yaml.full_load(f)

    general_config = config['general_config']
    network_config = config['CNN_model']

    Module = BasicModule('CNN', general_config, network_config=network_config)

    data_file = os.path.join('../data', 'dim_pro8_single_try.mat')
    train_dataset = HeatDataset(data_file, shuffle=False)
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=128, shuffle=False, drop_last=False)
    HT_computer = Characteristic()

    Nu = train_dataset.data.target[:, 0]
    Fan = train_dataset.data.target[:, 1]

    Nu_Fa = []

    for data in train_loader:
        design, coords, fields, target = data

        Nu_Fa.append(HT_computer(fields, coords, design).cpu())

    Nu_Fa = paddle.concat(Nu_Fa, axis=0).numpy()
    import matplotlib.pyplot as plt
    import visual_data as visual

    logger = visual.MatplotlibVision("\\")

    fig, axs = plt.subplots(2, 1, figsize=(15, 8), num=1)
    plt.subplot(211)
    logger.plot_regression(fig, axs[0], Nu, Nu_Fa[:, 0])

    plt.subplot(212)
    logger.plot_regression(fig, axs[1], Fan, Nu_Fa[:, 1])

    plt.show()
