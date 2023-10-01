# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/9/30 10:04
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：CNN_model.py
@File ：CNN_model.py
"""
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from utilize import activation_dict

class Identity(nn.Layer):
    '''
    a placeholder layer similar to tensorflow.no_op():
    https://github.com/pypaddle/pypaddle/issues/9160#issuecomment-483048684
    not used anymore as
    https://pypaddle.org/docs/stable/generated/paddle.nn.Identity.html
    edge and grid are dummy inputs
    '''

    def __init__(self, in_features=None, out_features=None,
                 *args, **kwargs):
        super(Identity, self).__init__()

        if in_features is not None and out_features is not None:
            self.id = nn.Linear(in_features, out_features)

        else:
            self.id = nn.Identity()

    def forward(self, x, edge=None, grid=None):
        """
        forward compute
        :param in_var: (batch_size, input_dim, ...)
        """
        # todo: 利用 einsteinsum 构造
        if len(x.shape) == 5:
            '''
            (-1, in, H, W, S) -> (-1, out, H, W, S)
            Used in SimpleResBlock
            '''
            x = x.transpose([0, 2, 3, 4, 1])
            x = self.id(x)
            x = x.transpose([0, 4, 1, 2, 3])
        elif len(x.shape) == 4:
            '''
            (-1, in, H, W) -> (-1, out, H, W)
            Used in SimpleResBlock
            '''
            x = x.transpose([0, 2, 3, 1])
            x = self.id(x)
            x = x.transpose([0, 3, 1, 2])

        elif len(x.shape) == 3:
            '''
            (-1, S, in) -> (-1, S, out)
            Used in SimpleResBlock
            '''
            # x = x.transpose([0, 2, 1])
            x = self.id(x)
            # x = x.transpose([0, 2, 1])
        elif len(x.shape) == 2:
            '''
            (-1, in) -> (-1, out)
            Used in SimpleResBlock
            '''
            x = self.id(x)
        else:
            raise NotImplementedError("input sizes not implemented.")

        return x



class Conv2dResBlock(nn.Layer):
    '''
    Conv2d + a residual block
    https://github.com/pypaddle/vision/blob/master/paddlevision/models/resnet.py
    Modified from ResNet's basic block, one conv less, no batchnorm
    No batchnorm
    '''

    def __init__(self, in_dim, out_dim,
                 kernel_size=3,
                 padding=1,
                 dilation=1,
                 dropout=0.1,
                 stride=1,
                 bias=False,
                 residual=False,
                 activation='silu',
                 basic_block=False,
                 ):
        super(Conv2dResBlock, self).__init__()

        self.activation = activation_dict[activation]
        self.add_res = residual
        self.conv = nn.Sequential(
            nn.Conv2D(in_dim, out_dim,
                      kernel_size=kernel_size,
                      padding=padding,
                      dilation=dilation,
                      stride=stride,
                      bias_attr=bias),
            nn.Dropout(dropout),
        )
        self.basic_block = basic_block
        if self.basic_block:
            self.conv1 = nn.Sequential(
                self.activation,
                nn.Conv2D(out_dim, out_dim,
                          kernel_size=kernel_size,
                          padding=padding,
                          bias_attr=bias),
                nn.Dropout(dropout),
            )
        self.apply_shortcut = (in_dim != out_dim)

        if self.add_res:
            if self.apply_shortcut:
                self.res = Identity(in_dim, out_dim)
            else:
                self.res = Identity()

    def forward(self, x):
        """
        forward compute
        :param in_var: (batch_size, input_dim, H, W)
        """
        if self.add_res:
            h = self.res(x)

        x = self.conv(x)

        if self.basic_block:
            x = self.conv1(x)

        if self.add_res:
            return self.activation(x + h)
        else:
            return self.activation(x)


class DeConv2dBlock(nn.Layer):
    '''
    Similar to a LeNet block
    4x upsampling, dimension hard-coded
    '''

    def __init__(self, in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 stride: int = 2,
                 kernel_size: int = 2,
                 # padding: int = 2,
                 # output_padding: int = 1,
                 dropout=0.1,
                 activation='silu'):
        super(DeConv2dBlock, self).__init__()
        # assert stride*2 == scaling_factor
        # padding1 = padding // 2 if padding // 2 >= 1 else 1

        self.deconv0 = nn.Conv2DTranspose(in_channels=in_dim,
                                          out_channels=hidden_dim,
                                          kernel_size=kernel_size,
                                          stride=stride
                                          # output_padding=output_padding,
                                          # padding=padding
                                          )
        self.conv0 = nn.Conv2D(in_channels=hidden_dim,
                               out_channels=out_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        # self.deconv1 = nn.ConvTranspose2d(in_channels=hidden_dim,
        #                                   out_channels=out_dim,
        #                                   kernel_size=kernel_size,
        #                                   stride=stride,
        #                                   # output_padding=output_padding,
        #                                   # padding=padding1,  # hard code bad, 1: for 85x85 grid, 2 for 43x43 grid
        #                                   )
        self.activation = activation_dict[activation]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        forward compute
        :param in_var: (batch_size, input_dim, H, W)
        """
        x = self.deconv0(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.conv0(x)
        # x = self.deconv1(x)
        # x = self.activation(x)
        return x


class Interp2dUpsample(nn.Layer):
    '''
    interpolate then Conv2dResBlock
    old code uses lambda and cannot be pickled
    temp hard-coded dimensions
    '''

    def __init__(self, in_dim: int,
                 out_dim: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 residual=False,
                 conv_block=True,
                 interp_mode='bilinear',
                 interp_size=None,
                 activation='silu',
                 dropout=0.1):
        super(Interp2dUpsample, self).__init__()
        self.activation = activation_dict[activation]
        self.dropout = nn.Dropout(dropout)
        if conv_block:
            self.conv = nn.Sequential(Conv2dResBlock(
                in_dim, out_dim,
                kernel_size=kernel_size,
                padding=padding,
                residual=residual,
                dropout=dropout,
                activation=activation),
                self.dropout,
                self.activation)
        self.conv_block = conv_block
        self.interp_size = interp_size
        self.interp_mode = interp_mode

    def forward(self, x):
        """
        forward compute
        :param in_var: (batch_size, input_dim, H, W)
        """
        if self.conv_block:
            x = self.conv(x)

        x = F.interpolate(x, size=self.interp_size,
                          mode=self.interp_mode,
                          align_corners=True)
        return x


class UNet2d(nn.Layer):
    """
        2维UNet
    """

    def __init__(self, in_sizes: tuple, out_sizes: tuple, width=32, depth=4, steps=1, activation='gelu',
                 dropout=0.0):
        """
        :param in_sizes: (H_in, W_in, C_in)
        :param out_sizes: (H_out, W_out, C_out)
        :param width: hidden dim, int
        :param depth: hidden layers, int
        """
        super(UNet2d, self).__init__()

        self.in_sizes = in_sizes[:-1]
        self.out_sizes = out_sizes[:-1]
        self.in_dim = in_sizes[-1]
        self.out_dim = out_sizes[-1]
        self.width = width
        self.depth = depth
        self.steps = steps

        self._input_sizes = [0, 0]
        self._input_sizes[0] = max(2 ** math.floor(math.log2(self.in_sizes[0])), 2 ** depth)
        self._input_sizes[1] = max(2 ** math.floor(math.log2(self.in_sizes[1])), 2 ** depth)


        self.interp_in = Interp2dUpsample(in_dim=steps*self.in_dim + 2, out_dim=self.in_dim, activation=activation,
                                          dropout=dropout, interp_size=self._input_sizes, conv_block=True)
        self.encoders = nn.LayerList()
        for i in range(self.depth):
            if i == 0:
                self.encoders.append(
                    Conv2dResBlock(self.in_dim, width, basic_block=True, activation=activation, dropout=dropout))
            else:
                self.encoders.append(nn.Sequential(nn.MaxPool2D(2),
                                                   Conv2dResBlock(2 ** (i - 1) * width, 2 ** i * width,
                                                                  basic_block=True, activation=activation,
                                                                  dropout=dropout)))

        self.bottleneck = nn.Sequential(nn.MaxPool2D(2),
                                        Conv2dResBlock(2 ** i * width, 2 ** i * width * 2, basic_block=True,
                                                       activation=activation, dropout=dropout))

        self.decoders = nn.LayerList()
        self.upconvs = nn.LayerList()

        for i in range(self.depth, 0, -1):
            self.decoders.append(
                Conv2dResBlock(2 ** i * width, 2 ** (i - 1) * width, activation=activation,
                               basic_block=True, dropout=dropout))
            self.upconvs.append(
                DeConv2dBlock(2 ** i * width, 2 ** (i - 1) * width, 2 ** (i - 1) * width, activation=activation,
                              dropout=dropout))

        self.conv1 = Conv2dResBlock(in_dim=width, out_dim=self.out_dim, basic_block=False, activation=activation,
                                    dropout=dropout)

        self.interp_out = Interp2dUpsample(in_dim=self.out_dim, out_dim=self.out_dim, interp_size=self.out_sizes,
                                           conv_block=False, activation=activation, dropout=dropout)

        self.conv2 = nn.Conv2D(self.out_dim, self.out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, grid):
        """
        forward computation
        """
        if len(x.shape) != len(grid.shape):
            repeat_times = paddle.to_tensor([1]+grid.shape[1:-1]+[1], dtype='int32')
            x = paddle.tile(x[:, None, None, :], repeat_times=repeat_times)

        x = paddle.concat((x, grid), axis=-1)
        x = x.transpose([0, 3, 1, 2])
        enc = []
        enc.append(self.interp_in(x))
        for i in range(self.depth):
            enc.append(self.encoders[i](enc[-1]))

        x = self.bottleneck(enc[-1])

        for i in range(self.depth):
            x = self.upconvs[i](x)
            x = paddle.concat((x, enc[-i - 1]), axis=1)
            x = self.decoders[i](x)

        x = self.interp_out(self.conv1(x))
        x = self.conv2(x)
        return x.transpose([0, 2, 3, 1])

if __name__ == '__main__':

    x = paddle.ones([10, 8])
    g = paddle.ones([10, 4, 92, 2])
    input_size = g.shape[1:-1] + x.shape[-1:]
    layer = UNet2d(in_sizes=input_size, out_sizes=(32, 32, 5), width=32, depth=6, steps=1)
    y = layer(x, g)
    print(y.shape)

