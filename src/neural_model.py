# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/9/30 15:09
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：neural_model.py
@File ：neural_model.py
"""

import os, sys, random, math
import datetime, time

import pandas as pd

from process_data import HeatDataset
from utilize import makeDirs, activation_dict, lossfunc_dict

import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from visual_data import MatplotlibVision
from utilize import LogHistory


class BasicModule(object):

    def __init__(self, name, config, network_config, load_path=None):

        self.name = name
        if load_path is None or not os.path.isdir(load_path):
            self.config = config
            self.network_config = network_config
            self._initialize()
        else:
            self._load(load_path)

        self.characteristic = Characteristic()
        # 物理场采用二阶范数
        self.fields_metric = PhysicsLpLoss(p=2, samples_reduction=False, channel_reduction=False)
        # 性能参数使用一阶范数, 同时考虑到f有接近于0的情况，使用relative=False
        self.target_metric = PhysicsLpLoss(p=1, relative=False, samples_reduction=False, channel_reduction=False)


    def train(self, train_loader, valid_loader):

        for epoch in range(1, self.total_epoch + 1):

            sta_time = time.time()
            self.train_epoch(train_loader)
            train_epoch_time = time.time() - sta_time

            if epoch % self.print_freq == 0:
                train_metric, train_loss = self.valid_epoch(train_loader)
                valid_train_time = time.time() - train_epoch_time - sta_time
                valid_metric, valid_loss = self.valid_epoch(valid_loader)
                valid_valid_time = time.time() - valid_train_time - train_epoch_time - sta_time


                self.loghistory.append(epoch=epoch,
                                       time_train=train_epoch_time,
                                       time_valid=valid_train_time + valid_valid_time,
                                       loss_train=train_loss['fields'], #训练总损失
                                       loss_valid=valid_loss['fields'], #验证总损失
                                       loss_target_train=train_loss['target'],  # Nusselt 和 Fanning 训练总损失
                                       loss_target_valid=valid_loss['target'],  # Nusselt 和 Fanning 验证总损失
                                       metric_train=train_metric['fields'],  # 各个物理场训练损失
                                       metric_valid=valid_metric['fields'],  # 各个物理场验证损失
                                       metric_target_train=train_metric['target'],
                                       metric_target_valid=valid_metric['target'])

                print('epoch: {:6d}, learning_rate: {:.3e}, '
                      'train_cost: {:.2f}, valid_cost: {:.2f}, '
                      'train_epoch_loss: {:.3e}, valid_epoch_loss: {:.3e}, '
                      'train_target_loss: {:.3e}, valid_target_loss: {:.3e}'.
                      format(epoch, self.optimizer.get_lr(),
                             self.loghistory.time_train[-1], self.loghistory.time_valid[-1],
                             self.loghistory.loss_train[-1].mean(), self.loghistory.loss_valid[-1].mean(),
                             self.loghistory.loss_target_train[-1].mean(), self.loghistory.loss_target_valid[-1].mean()
                             ))


                fig, axs = plt.subplots(2, 1, figsize=(15, 8), num=1, constrained_layout=True)
                self.visual.plot_loss(fig, axs[0], label='train_loss',
                                      x=np.array(self.loghistory.epoch_list),
                                      y=np.array(self.loghistory.loss_train).mean(axis=-1),
                                      std=np.array(self.loghistory.loss_train).std(axis=-1))
                self.visual.plot_loss(fig, axs[0], label='valid_loss',
                                      x=np.array(self.loghistory.epoch_list),
                                      y=np.array(self.loghistory.loss_valid).mean(axis=-1),
                                      std=np.array(self.loghistory.loss_valid).std(axis=-1))
                self.visual.plot_loss(fig, axs[1], label='train_target_loss',
                                      x=np.array(self.loghistory.epoch_list),
                                      y=np.array(self.loghistory.loss_target_train).mean(axis=-1),
                                      std=np.array(self.loghistory.loss_target_train).std(axis=-1))
                self.visual.plot_loss(fig, axs[1], label='valid_target_loss',
                                      x=np.array(self.loghistory.epoch_list),
                                      y=np.array(self.loghistory.loss_target_valid).mean(axis=-1),
                                      std=np.array(self.loghistory.loss_target_valid).std(axis=-1))

                fig.suptitle('training process')
                fig.savefig(os.path.join(self.train_path, 'training_process.jpg'), dpi=600)
                plt.close(fig)

                pd.DataFrame(np.concatenate((
                                             np.array(self.loghistory.loss_train).mean(axis=-1),
                                             np.array(self.loghistory.loss_train).std(axis=-1),
                                             np.array(self.loghistory.loss_valid).mean(axis=-1),
                                             np.array(self.loghistory.loss_valid).std(axis=-1),
                                             np.array(self.loghistory.loss_target_train).mean(axis=-1),
                                             np.array(self.loghistory.loss_target_train).std(axis=-1),
                                             np.array(self.loghistory.loss_target_valid).mean(axis=-1),
                                             np.array(self.loghistory.loss_target_valid).std(axis=-1)),axis=-1).reshape(-1, 8),
                             columns=['train_loss_mean', 'train_loss_std', 'valid_loss_mean',
                                      'valid_loss_std', 'train_target_loss_mean', 'train_target_loss_std',
                                      'valid_target_loss_mean','valid_target_loss_std']).to_csv(
                    os.path.join(self.train_path, 'training.csv'))


                fig, axs = plt.subplots(2, 1, figsize=(15, 8), num=2, constrained_layout=True)
                labels = ['valid_loss_mean_' + name for name in self.visual.field_name]
                self.visual.plot_loss(fig, axs[0], label=labels,
                                      x=np.array(self.loghistory.epoch_list),
                                      y=np.array(self.loghistory.metric_valid).mean(axis=1))
                labels = ['valid_loss_max_' + name for name in self.visual.field_name]
                self.visual.plot_loss(fig, axs[1], label=labels,
                                      x=np.array(self.loghistory.epoch_list),
                                      y=np.array(self.loghistory.metric_valid).max(axis=1))
                fig.suptitle('fields_valid_process')
                fig.savefig(os.path.join(self.train_path, 'fields_valid.jpg'), dpi=600)
                plt.close(fig)

                pd.DataFrame(np.concatenate((
                                             np.array(self.loghistory.metric_valid).mean(axis=1),
                                             np.array(self.loghistory.metric_valid).max(axis=1)), axis=-1),
                             columns=['p_mean', 't_mean', 'u_mean','v_mean',
                                      'p_max', 't_max', 'u_max','v_max']).to_csv(
                    os.path.join(self.train_path, 'fields.csv'))



                fig, axs = plt.subplots(2, 1, figsize=(15, 8), num=3, constrained_layout=True)
                self.visual.plot_loss(fig, axs[0], label=['valid_loss_mean_nu', 'valid_loss_mean_f'],
                                       x=np.array(self.loghistory.epoch_list),
                                       y=np.array(self.loghistory.metric_target_valid).mean(axis=1))
                self.visual.plot_loss(fig, axs[1], label=['valid_loss_max_nu', 'valid_loss_max_f'],
                                       x=np.array(self.loghistory.epoch_list),
                                       y=np.array(self.loghistory.metric_target_valid).max(axis=1))
                fig.suptitle('target_valid_process')
                fig.savefig(os.path.join(self.train_path, 'target_valid.jpg'), dpi=600)
                plt.close(fig)

                pd.DataFrame(np.concatenate((
                                             np.array(self.loghistory.metric_target_valid).mean(axis=1),
                                             np.array(self.loghistory.metric_target_valid).max(axis=1)),
                                            axis=-1),
                             columns=['nu_mean', 'nu_max', 'f_mean', 'f_max']).to_csv(
                    os.path.join(self.train_path, 'loss_target.csv'))

            if epoch % self.save_freq == 0:
                paddle.save({
                    'epoch': epoch,
                    'config': self.config,
                    'network_config': self.network_config,
                    'network': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()},
                    os.path.join(self.work_path, 'last_model.pdparams'))

                self.loghistory.save(os.path.join(self.work_path, 'loghistory.pkl'))

    def infer(self, data_loader, data_name, show_nums=20):

        all_fields_true = []
        all_fields_pred = []
        all_target_true = []
        all_target_pred = []
        all_design, all_coords = [], []

        with paddle.no_grad():
            for data in data_loader:
                design, coords, fields_true, _ = data

                fields_pred = self.network(design, coords)

                design = data_loader.design_back(design)
                coords = data_loader.coords_back(coords)
                fields_true = data_loader.fields_back(fields_true)
                fields_pred = data_loader.fields_back(fields_pred)

                target_true = self.characteristic(fields_true, coords, design)
                target_pred = self.characteristic(fields_pred, coords, design)

                all_coords.append(coords.cpu().numpy())
                all_design.append(design.cpu().numpy())
                all_fields_true.append(fields_true.cpu().numpy())
                all_fields_pred.append(fields_pred.cpu().numpy())
                all_target_true.append(target_true.cpu().numpy())
                all_target_pred.append(target_pred.cpu().numpy())

        all_coords = np.concatenate(all_coords, axis=0)
        all_design = np.concatenate(all_design, axis=0)
        all_target_true = np.concatenate(all_target_true, axis=0)
        all_target_pred = np.concatenate(all_target_pred, axis=0)
        all_fields_true = np.concatenate(all_fields_true, axis=0)
        all_fields_pred = np.concatenate(all_fields_pred, axis=0)

        np.save(os.path.join(self.infer_path, data_name + '_true.npy'), all_fields_true)
        np.save(os.path.join(self.infer_path, data_name + '_pred.npy'),  all_fields_pred)
        np.save(os.path.join(self.infer_path, data_name + '.npy'), all_coords)

        err_target = ((all_target_pred - all_target_true)
                      / (np.max(all_target_true, axis=0) - np.min(all_target_true, axis=0)))
        mse_target =  ((all_target_pred - all_target_true) / all_target_true) * 100

        self.save_path = os.path.join(self.infer_path, data_name)
        makeDirs(self.save_path)

        pd.DataFrame(np.concatenate((all_target_true, all_target_pred, mse_target), axis=-1),
                     columns=['Nu_true', 'f_true',  'Nu_pred', 'f_pred', 'nu_error' , 'f_error']).to_csv(
            os.path.join(self.save_path, 'target.csv'))

        # for epoch in range(1, self.total_epoch + 1):
        #     if epoch % 200 == 0:
                # for tar_id in range(all_target_true.shape[-1]):
        fig, axs = plt.subplots(2, 2, figsize=(32, 16), num=1, constrained_layout=True)
        self.visual.plot_regression(fig, axs[0, 0], all_target_true[:, 0], all_target_pred[:, 0],
                                    title='Nu')
        self.visual.plot_regression(fig, axs[0, 1], all_target_true[:, 1], all_target_pred[:, 1],
                                    title='f')

        self.visual.plot_error(fig, axs[1, 0], err_target[:, 0], title='Nu error')
        self.visual.plot_error(fig, axs[1, 1], err_target[:, 1], title='f error')
        fig.savefig(os.path.join(self.infer_path, data_name, 'target_pred.jpg'),
                    dpi=600, bbox_inches='tight')
        plt.close(fig)

        for fig_id in range(show_nums):
            fig, axs = plt.subplots(4, 3, figsize=(32, 16), num=2, constrained_layout=True)
            axs_flat = axs.flatten()
            for ax in axs_flat:
                ax.axis('off')
                ax.set_frame_on(False)
            self.visual.plot_fields_ms(fig, axs, all_fields_true[fig_id], all_fields_pred[fig_id],
                                       all_coords[fig_id])
            fig.savefig(os.path.join(self.infer_path, data_name, 'solution_whole_' + str(fig_id) + '.jpg'),
                        dpi=600, bbox_inches='tight')
            plt.close(fig)

            fig, axs = plt.subplots(4, 3, figsize=(32, 16), num=3)
            axs_flat = axs.flatten()
            for ax in axs_flat:
                ax.axis('off')
                ax.set_frame_on(False)
            self.visual.plot_fields_ms(fig, axs, all_fields_true[fig_id], all_fields_pred[fig_id],
                                       all_coords[fig_id],
                                       cmin_max=[[0.0010, 0.00008], [0.0025, 0.00042]])
            fig.savefig(os.path.join(self.infer_path, data_name, 'solution_local_' + str(fig_id) + '.jpg'),
                        dpi=600, bbox_inches='tight')
            plt.close(fig)

    #  训练函数
    def train_epoch(self, train_loader):

        self.network.train()
        for data in train_loader:
            design, coords, fields, _ = data
            self.optimizer.clear_grad()
            fields_ = self.network(design, coords)
            loss = self.loss_func(fields_, fields)
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()


# 验证函数
    def valid_epoch(self, data_loader):

        log_metric = {'target': [], 'fields': []}
        log_loss = {'target': [], 'fields': []}

        self.network.eval()
        with paddle.no_grad():
            for data in data_loader:
                design, coords, fields, _ = data

                fields_ = self.network(design, coords)
                fields_loss = self.loss_func(fields_, fields).item()

                design = data_loader.design_back(design)
                coords = data_loader.coords_back(coords)
                fields = data_loader.fields_back(fields)
                fields_ = data_loader.fields_back(fields_)

                target = self.characteristic(fields, coords, design)
                target_ = self.characteristic(fields_, coords, design)

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



    def _initialize(self):

        self._set_config()
        self._set_path()
        self._set_device()
        self._set_network()
        self._set_optim()
        self._set_logger()
        self._set_visual()

    def _load(self, load_path):

        self._set_path(load_path)
        try:
            model_state = paddle.load(os.path.join(self.work_path, 'last_model.pdparams'))
        except:
            raise ValueError("the last_model.pdparams doesnt exist!")

        try:
            self.config = model_state['config']
            self._set_config()
        except:
            raise ValueError("the config format is not correct!")

        try:
            self.network_config = model_state['network_config']
        except:
            raise ValueError("the network_config format is not correct!")

        self._set_device()
        self._set_network()
        self._set_optim()
        self._set_visual()
        self._set_logger()

        try:
            load_network = model_state['network']
            load_optimizer = model_state['optimizer']
            load_scheduler = model_state['scheduler']
            self.network.set_state_dict(load_network)
            self.optimizer.set_state_dict(load_optimizer)
            self.scheduler.set_state_dict(load_scheduler)
        except:
            raise ValueError("the netmodel can't be loaded!")

        # 记录读取
        try:
            self.loghistory.load(os.path.join(self.work_path, 'loghistory.pkl'))
        except:
            raise ValueError("the loghistory doesn't exist!")


    # def _load_config(self, load_config):
    #     self.config.update(load_config)
    #     self._set_config()

    def _set_config(self):
        all_keys = list(self.config.keys())
        for key in all_keys:
            setattr(self, key, self.config[key])

    def _set_device(self):

        if paddle.device.is_compiled_with_cuda():
            device = paddle.device.set_device('gpu')
        else:
            device = paddle.device.set_device('cpu')
        self.device = device

        print('device: {}'.format(self.device))

    def _set_path(self, path=None):

        if path is None:
            current_datetime = datetime.datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
            self.work_path = os.path.join(self.root_path, 'work', self.name, formatted_datetime)
        else:
            self.work_path = path

        self.train_path = os.path.join(self.work_path, 'train')
        self.valid_path = os.path.join(self.work_path, 'valid')
        self.infer_path = os.path.join(self.work_path, 'infer')

        makeDirs([self.work_path, self.train_path, self.valid_path, self.infer_path])
        print('work path is : {}'.format(self.work_path))


    def _set_network(self):
        if 'FNO' in self.name:
            from src.FNO_model import FNO2d
            self.network = FNO2d(**self.network_config)
        elif 'CNN' in self.name:
            from src.CNN_model import UNet2d
            self.network = UNet2d(**self.network_config)
        elif 'DON' in self.name:
            from src.DON_model import DeepONetMulti
            self.network = DeepONetMulti(**self.network_config)
        elif 'MLP' in self.name:
            from src.DON_model import FcnMulti
            self.network = FcnMulti(**self.network_config)
        elif 'TNO' in self.name:
            from src.TNO_model import FourierTransformer2D
            self.network = FourierTransformer2D(**self.network_config)
        self.network = self.network.to(self.device)
        print('network name is : {}'.format(self.name))

    def _set_optim(self):

        model_parameters = filter(lambda p: ~p.stop_gradient, self.network.parameters())
        params = sum([np.prod(p.shape) for p in model_parameters])
        print("Initialized {} with {} trainable params ".format(self.name, params))

        self.loss_func = lossfunc_dict[self.loss_name]

        self.scheduler = optim.lr.MultiStepDecay(learning_rate=self.learning_rate,
                                                 milestones=self.learning_milestones,
                                                 gamma=self.learning_gamma)

        self.optimizer = optim.Adam(parameters=self.network.parameters(),
                                    learning_rate=self.scheduler,
                                    beta1=self.learning_beta[0], beta2=self.learning_beta[1],
                                    weight_decay=self.weight_decay)

    def _set_logger(self):

        self.loghistory = LogHistory(log_names=('loss_target_train', 'loss_target_valid',
                                                'metric_target_train', 'metric_target_valid'))

    def _set_visual(self):

        self.visual = MatplotlibVision(self.work_path, input_name=('x', 'y'), field_name=('P', 'T', 'U', 'V'))

#积分计算性能参数 Nusselt 和 Fanning
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
        h = hflex / (paddle.abs(Tw - Tb) + 1e-8)  # 保证数值稳定性
        Nu = h * D / lamda

        vel = Re * miu / rho / D
        Dp = self.cal_f(X, Y, P)
        Fan = Dp * D / 2 / L / vel / vel / rho

        result = paddle.stack((Nu, Fan), axis=1)

        return result


# 物理场及性能参数的相对范数计算 ：物理场 L2, 性能参数 L1
class PhysicsLpLoss(object):
    def __init__(self, p=2, relative=True, samples_reduction=True, channel_reduction=False):
        super(PhysicsLpLoss, self).__init__()

        # Lp-norm type are positive
        assert p > 0, 'Lp-norm type should be positive!'

        self.p = p
        self.relative = relative
        self.channel_reduction = channel_reduction
        self.samples_reduction = samples_reduction

    def forward(self, x, y):     #x预测场； y真实场

        if paddle.is_tensor(x):
            dif_norms = paddle.norm(x.reshape((x.shape[0], -1, x.shape[-1])) -
                                    y.reshape((x.shape[0], -1, x.shape[-1])), self.p, 1)
            all_norms = paddle.norm(y.reshape((x.shape[0], -1, x.shape[-1])), self.p, 1)

            if self.relative:
                res_norms = dif_norms / (all_norms + 1e-8)
            else:
                res_norms = dif_norms

            if self.samples_reduction:
                res_norms = paddle.mean(res_norms, axis=0)    #在样本角度平均

            if self.channel_reduction:
                res_norms = paddle.mean(res_norms, axis=-1)   #在物理场角度平均

        else:
            dif_norms = np.linalg.norm(x.reshape(x.shape[0], -1, x.shape[-1]) -
                                       y.reshape(x.shape[0], -1, x.shape[-1]), self.p, 1)
            all_norms = np.linalg.norm(y.reshape(x.shape[0], -1, x.shape[-1]), self.p, 1)

            if self.relative:
                res_norms = dif_norms / (all_norms + 1e-8)
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

    with open(os.path.join('../default_config.yml')) as f:
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

    import visual_data as visual

    logger = visual.MatplotlibVision("\\")

    fig, axs = plt.subplots(2, 1, figsize=(15, 8), num=1)
    plt.subplot(211)
    logger.plot_regression(fig, axs[0], Nu, Nu_Fa[:, 0])

    plt.subplot(212)
    logger.plot_regression(fig, axs[1], Fan, Nu_Fa[:, 1])

    plt.show()
