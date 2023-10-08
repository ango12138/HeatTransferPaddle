# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/9/30 10:14
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：process_data.py
@File ：process_data.py
"""

import os.path
import numpy as np
import scipy.io as sio
import h5py
import sklearn.metrics
import paddle
from paddle.io import Dataset, DataLoader

# reading data
class MatLoader(object):
    def __init__(self, file_path, to_paddle=True, to_cuda=False, to_float=True):
        super(MatLoader, self).__init__()

        self.to_paddle = to_paddle
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):

        try:
            self.data = sio.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_paddle:
            x = paddle.to_tensor(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_paddle(self, to_paddle):
        self.to_paddle = to_paddle

    def set_float(self, to_float):
        self.to_float = to_float



class DataNormer(object):
    """
        data normalization at last dimension
    """

    def __init__(self, data, method="min-max", axis=None):
        """
            data normalization at last dimension
            :param data: data to be normalized
            :param method: normalization method
            :param axis: axis to be normalized
        """
        if isinstance(data, str):
            if os.path.isfile(data):
                try:
                    self.load(data)
                except:
                    raise ValueError("the savefile format is not supported!")
            else:
                raise ValueError("the file does not exist!")
        elif type(data) is np.ndarray:
            if axis is None:
                axis = tuple(range(len(data.shape) - 1))
            self.method = method
            if method == "min-max":
                self.max = np.max(data, axis=axis)
                self.min = np.min(data, axis=axis)

            elif method == "mean-std":
                self.mean = np.mean(data, axis=axis)
                self.std = np.std(data, axis=axis)
        elif type(data) is paddle.Tensor:
            if axis is None:
                axis = tuple(range(len(data.shape) - 1))
            self.method = method
            if method == "min-max":
                self.max = np.max(data.numpy(), axis=axis)
                self.min = np.min(data.numpy(), axis=axis)

            elif method == "mean-std":
                self.mean = np.mean(data.numpy(), axis=axis)
                self.std = np.std(data.numpy(), axis=axis)
        else:
            raise NotImplementedError("the data type is not supported!")


    def norm(self, x):
        """
            input tensors
            param x: input tensors
            return x: output tensors
        """
        if paddle.is_tensor(x):
            if self.method == "min-max":
                x = 2 * (x - paddle.to_tensor(self.min, place=x.place)) \
                    / (paddle.to_tensor(self.max, place=x.place) - paddle.to_tensor(self.min, place=x.place) + 1e-10) - 1
            elif self.method == "mean-std":
                x = (x - paddle.to_tensor(self.mean, place=x.place)) / (paddle.to_tensor(self.std + 1e-10, place=x.place))
        else:
            if self.method == "min-max":
                x = 2 * (x - self.min) / (self.max - self.min + 1e-10) - 1
            elif self.method == "mean-std":
                x = (x - self.mean) / (self.std + 1e-10)

        return x

    def back(self, x):
        """
            input tensors
            param x: input tensors
            return x: output tensors
        """
        if paddle.is_tensor(x):
            if self.method == "min-max":
                x = (x + 1) / 2 * (paddle.to_tensor(self.max)
                                   - paddle.to_tensor(self.min) + 1e-10) + paddle.to_tensor(self.min)
            elif self.method == "mean-std":
                x = x * (paddle.to_tensor(self.std + 1e-10)) + paddle.to_tensor(self.mean)
        else:
            if self.method == "min-max":
                x = (x + 1) / 2 * (self.max - self.min + 1e-10) + self.min
            elif self.method == "mean-std":
                x = x * (self.std + 1e-10) + self.mean
        return x
    def save(self, save_path):
        """
            save the parameters to the file
            :param save_path: file path to save
        """
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, save_path):
        """
            load the parameters from the file
            :param save_path: file path to load
        """
        import pickle
        isExist = os.path.exists(save_path)
        if isExist:
            try:
                with open(save_path, 'rb') as f:
                    load = pickle.load(f)
                self.method = load.method
                if load.method == "mean-std":
                    self.std = load.std
                    self.mean = load.mean
                elif load.method == "min-max":
                    self.min = load.min
                    self.max = load.max
            except:
                raise ValueError("the savefile format is not supported!")
        else:
            raise ValueError("The pkl file is not exist, CHECK PLEASE!")




class HeatDataset(Dataset):
    TRAIN = 0
    VALID = 1
    TEST = 2
    def __init__(self,
                 file,
                 mode=0,
                 sampler={'sample_mode': 'all', 'sample_size': 0.5},
                 shuffle=True,
                 training_size=0.8,
                 test_size=0.1):
        """
        :param dataProp: for split&mix from multiple dirs, see LoaderNormalizer; None means off
        :param mode: TRAIN|TEST , toggle regular 80/20 split for training & validation data, or load test data
        :param dataDir: directory containing training data
        :param dataDirTest: second directory containing test data , needs training dir for normalization
        :param normMode: toggle normalization
        """

        self.file = file
        self.mode = mode                   # only for mode==self.TEST
        self.sampler = sampler
        self.shuffle = shuffle
        self.training_size = training_size
        self.test_size = test_size

        self.data_read()
        self.data_norm()
        self.data_split()

    def data_read(self):

        class data:
            pass

        reader = MatLoader(self.file, to_paddle=False, to_cuda=False, to_float=True)
        self.data = data
        self.data.length = reader.read_field('data').shape[0]
        if not self.shuffle:
            self.shuffle_idx = np.arange(self.data.length)
        else:
            self.shuffle_idx = np.random.permutation(self.data.length)

        self.data.design = reader.read_field('data')[self.shuffle_idx]
        self.data.coords = reader.read_field('grids')[:, ::2, :, :2][self.shuffle_idx]   # 注意原始数据在x方向分辨率降低了1倍
        self.data.fields = reader.read_field('field')[:, ::2, :, :][self.shuffle_idx]
        self.data.target = np.concatenate((reader.read_field('Nu'), reader.read_field('f')), axis=-1)[self.shuffle_idx]


    def data_split(self):

        self.train_len = int(self.data.length * self.training_size)
        self.valid_len = int(self.data.length * (1 - self.training_size - self.test_size))
        self.test_len = self.data.length - self.train_len - self.valid_len

        if self.mode == self.TRAIN:
            self.data.design = self.data.design[:self.train_len]
            self.data.coords = self.data.coords[:self.train_len]
            self.data.fields = self.data.fields[:self.train_len]
            self.data.target = self.data.target[:self.train_len]
            self.data.length = self.train_len
        elif self.mode == self.VALID:
            self.data.design = self.data.design[self.train_len:self.train_len + self.valid_len]
            self.data.coords = self.data.coords[self.train_len:self.train_len + self.valid_len]
            self.data.fields = self.data.fields[self.train_len:self.train_len + self.valid_len]
            self.data.target = self.data.target[self.train_len:self.train_len + self.valid_len]
            self.data.length = self.valid_len
        elif self.mode == self.TEST:
            self.data.design = self.data.design[-self.test_len:]
            self.data.coords = self.data.coords[-self.test_len:]
            self.data.fields = self.data.fields[-self.test_len:]
            self.data.target = self.data.target[-self.test_len:]
            self.data.length = self.test_len

    def data_norm(self):

        class norm:
            pass
        self.norm = norm
        self.norm.design = DataNormer(self.data.design, method='min-max')
        self.norm.fields = DataNormer(self.data.fields, method='min-max')
        self.norm.coords = DataNormer(self.data.coords, method='min-max')
        self.norm.target = None

    def train(self):
        self._set_mode(0)
    def valid(self):
        self._set_mode(1)
    def test(self):
        self._set_mode(2)
    def _set_mode(self, mode_id):
        self.mode = mode_id

    def __len__(self):
        return self.data.length

    def __getitem__(self, idx):

        if self.mode == self.TRAIN:

            coords = self.data.coords[idx]
            fields = self.data.fields[idx]
            design = self.data.design[idx]
            target = self.data.target[idx]

            if self.sampler['sample_mode'] == 'random':
                coords = coords.reshape(-1, coords.shape[-1])
                fields = fields.reshape(-1, fields.shape[-1])
                sample_size = int(coords.shape[0] * self.sampler['sample_size'])
                sample_idx = np.random.choice(coords.shape[0], sample_size, replace=False)
                coords = coords[sample_idx]
                fields = fields[sample_idx]
            elif self.sampler['sample_mode'] == 'down':
                if isinstance(self.sampler['sample_size'], int):
                    coords = coords[::self.sampler['sample_size'], ::self.sampler['sample_size']]
                    fields = fields[::self.sampler['sample_size'], ::self.sampler['sample_size']]
                else:
                    coords = coords[::self.sampler['sample_size'][0], ::self.sampler['sample_size'][1]]
                    fields = fields[::self.sampler['sample_size'][0], ::self.sampler['sample_size'][1]]
            else:
                pass
        else:
            coords = self.data.coords[idx]
            fields = self.data.fields[idx]
            design = self.data.design[idx]
            target = self.data.target[idx]

        if self.norm.design is not None:
            design = self.norm.design.norm(design)
        if self.norm.fields is not None:
            fields = self.norm.fields.norm(fields)
        if self.norm.coords is not None:
            coords = self.norm.coords.norm(coords)
        if self.norm.target is not None:
            target = self.norm.target.norm(target)

        return design, coords, fields, target


class HeatDataLoader(DataLoader):

    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,):
        super(HeatDataLoader, self).__init__(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             drop_last=drop_last)

    def design_back(self, data):
        if self.dataset.norm.design is not None:
            return self.dataset.norm.design.back(data)
        else:
            return data

    def design_norm(self, data):
        if self.dataset.norm.design is not None:
            return self.dataset.norm.design.norm(data)
        else:
            return data

    def fields_back(self, data):
        if self.dataset.norm.fields is not None:
            return self.dataset.norm.fields.back(data)
        else:
            return data

    def fields_norm(self, data):
        if self.dataset.norm.fields is not None:
            return self.dataset.norm.fields.norm(data)
        else:
            return data

    def coords_back(self, data):
        if self.dataset.norm.coords is not None:
            return self.dataset.norm.coords.back(data)
        else:
            return data

    def coords_norm(self, data):
        if self.dataset.norm.coords is not None:
            return self.dataset.norm.coords.norm(data)
        else:
            return data

    def target_norm(self, data):
        if self.dataset.norm.target is not None:
            return self.dataset.norm.target.norm(data)
        else:
            return data

    def target_back(self, data):
        if self.dataset.norm.target is not None:
            return self.dataset.norm.target.back(data)
        else:
            return data