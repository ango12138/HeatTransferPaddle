# 基于深度学习的纳米流体物理场与性能预测研究-----技术文档



- 近年来，随着工业生产等领域的快速发展，大功率设备在运行过程中会产生大量的热量，导致设备性能下降。为了进一步提高传热能力，改善传热过程，采用纳米流体技术，利用掺杂纳米颗粒来提高流体的导热性能，从而增强冷却效果。实验测量对于理解纳米流体的流动和传热特性起着至关重要的作用，但由于测量技术的原因，缺乏详细的全局细节，这对该领域的实验研究造成了重大限制。因此，受到计算机视觉中的图像回归技术的启发，各种深度学习模型在工程中广泛应用于物理领域的建模。通过将完整的物理场视为在空间域中定义的图像，我们可以将不同的物理量概念化为该图像的不同RGB通道。这种类比使我们能够利用图像处理中使用的原则和方法来有效地分析和建模物理数据。

- 本文基于五种神经算子网络：FNO、U-Net、FNN、DeepONet和Transformer，对water-Al2O3纳米流体在微通道内的流动进行物理场预测，并从预测场中提取了表征流动换热性能的参数Nusselt数和Fanning摩擦因子。将二维物理场求解过程看作给定设计变量下的物理场回归任务：以微通道的8个设计变量及空间坐标为输入，输出4个物理场（压力、温度、速度场）。此外，我们还关注了工业中从物理场提取的实际性能参数的预测精度，采用积分求解的方式，实现对Nusselt数和Fanning摩擦因子的预测。
- 本次对五种神经算子网络的实现均基于Paddle，且所有模型训练在 AMD EPYC 7642  CPU 和 Nvidia A40 GPU 工作站进行。通过简单对比了五种网络预测的物理场误差，结果表明FNO和Transformer预测精度相对较高，U-Net和DeepONet的预测效果最差。



## 1. 代码说明



📂 Deep-Flow-Prediction-Paddle

|_📁 data                                                             #部分数据集

​     |_ 📄 dim_pro8_single_try.mat

|_📁 config

​     |_📄 CNN.yaml                                             # CNN的网络参数设置

​     |_📄 DNO.yaml                                            # DNO的网络参数设置

​     |_📄 FNO.yaml                                             # FNO的网络参数设置

​     |_📄 MLP.yaml                                             # FNN的网络参数设置

​     |_📄 TNO.yaml                                            # TNO的网络参数设置

|_📁 src

​    |_📄 process_data                                       # 读取matlab格式数据；数据归一化；划分数据集及数据采样

​    |_📄 CNN_model.py                                   # 二维U-Net model paddle代码

​    |_📄 DON_model.py                                  # 二维DeepONet 以及 FNN model paddle代码

​    |_📄 FNO_model.py                                   # 二维Fourier Neural Operator paddle代码

​    |_📄 TNO_model.py                                   # 二维Transformer paddle代码, 支持多种attention机制以及两种Regressor

​    |_📄 neural_model.py                               # 网络的训练及验证；性能参数的积分求解；可视化损失函数、物理场、性能参数

​    |_📄 process_data.py                                # 数据读取；数据归一化；数据集划分；

​    |_📄 utilize.py                                             # 激活函数；损失函数；初始化权重；记录训练信息

​    |_📄 visual_data.py                                    # 可视化代码

|_📁 work                                                       # 训练过程、验证结果、测试结果，统计结果文件保存

​    |_📁 DON                                                   # DeepONet训练结果保存

​         |_📁 2023-10-07-10-43                       # 以时间戳命名对结果保存

​                |_📁 infer                                       # 模型训练、验证、测试结果保存

​                |_📁 train                                      # 训练结果保存

​                |_📁 valid                                      # 验证结果保存

​                |_📄 last_model.pdparams        # 保存的模型文件

​                |_📄 loghistory.pkl                       # 保存的epoch、训练和预测时间、物理场及性能参数损失文件

​    |_📁 FNO                                                   # FNO训练结果保存

​    |_📁 CNN                                                  # U-Net训练结果保存

​    |_📁 TNO                                                  # Transformer训练结果保存

​    |_📁 FNN                                                  # FNN训练结果保存

|_📄 run_infer.py                                         # 测试过程

|_📄 run_main.py                                       # 训练过程、验证过程



### 1.1 训练参数设置

```python
basic_config:
  root_path: './'                        #读取数据路径
  data_name: 'dim_pro8_single_all.mat'   #数据集名称
  training_size: 0.8                     #训练集占样本的比例
  batch_size: 32                         #每次传递给网络用来训练的样本个数
  total_epoch: 500                       #训练总步长
  loss_name: 'mse'                       #损失函数类型
  learning_rate: 1.e-4                   #学习率
  weight_decay: 1.e-12                   #权重衰减
  learning_beta:                         #控制梯度信息的参数
    - 0.7
    - 0.9
  learning_milestones:                   #学习率下降节点
    - 300
    - 400
    - 500
  learning_gamma: 0.1                    #学习率下降倍数
```

### 1.2 快速运行

- config文件下的五个yaml脚本分别为五种神经算子网络的参数配置。其中，“basic_config”为训练参数的设置，”network_config“是模型参数的设置，training_size为训练集占总样本的比例，batch_size为单次传递给程序用以训练的数据 (样本) 个数，total_epoch为训练的总步长，loss_name为物理场或性能参数的总损失的类型；learning_rate、weight_decay、learning_beta、learning_milestones 和learning_gamma分别为学习率，权重衰减，Adam中控制梯度信息的超参数，学习率下降的节点，学习率每次下降的倍数。”name_model“为不同算子网络的具体参数设置。

- 运行run_main.py，对模型进行训练和验证。首先，确定想要训练的模型，在原始配置文件path处，输入目标模型的yaml文件。（config文件下的所有模型的yaml文件内的参数设置，均为我们训练好的参数，可直接调用）。其次，在网络参数配置处，输入不同的name会调用不同的网络模型，在work文件下生成对应名字的子文件，子文件下有以时间戳命名的训练结果。同时输出物理的总loss_train和loss_valid，性能参数的总损失loss_target_train和loss_target_valid，四个物理场的各自的损失metric_train和metric_valid，两个性能参数的各自损失metric_target_train和metric_target_valid。运行后会生成20个case的物理场云图、性能参数回归图，及各个训练过程的损失曲线。
- 运行run_infer.py，首先修改网络名称，输入对应模型的yaml文件路径。其次，把run_main.py中已经训练好并保存的模型文件（在work文件下的对应模型名称和时间戳下的last_model.pdparams）路径添加到load_path中，Module通过加载其路径，传递保存的config和network_config，调用BasicModule。通过输入测试集（即test_dataset中mode=1）实现对模型的快速预测。
- 考虑到运行时间过久，为了快速查看运行结果，可先设置basci_config中total_epoch=10， print_freq=1, save_freq=2，跑出一组查看效果，结果在work文件下对应算子网络名称的子文件的时间戳中保存。

### 1.3 环境依赖

>  numpy==1.26.0
>
>  cudnn==8.4.1.50
>
>  cudatoolkit==11.6.2
>
>  paddlepaddle-gpu develop 2.5.1.
>
>  pyyaml==6.0
>
>  scipy==1.11.3
>
>  scikit-learning==1.3.0
>
>  seaborn==0.12.2
>
>  matplotlib==3.7.2



## 2. 原始数据集

​        本研究中，我们采用商业软件ICEM CFD和FLUENT对网格进行划分以及数值计算，基于有限体积法，对控制方程进行时均化处理，所有方程均采用SIMPLE 算法结合二阶迎风离散化方案求解。当所有控制方程的残差小于 10-6，且相邻迭代之间平均壁温和压降的残差小于 0.1%时，认为数值模型收敛。

 		本文，以Al2O3 纳米流体在带有凹槽的微通道中流动的二维流动换热问题作为研究对象，验证物理场预测以及流动换热性能的识别方法。 鉴于其几何形状，可以在沿 z 方向的任何位置获得 x-y 平面模型，因此我们将三维模型简化为二维模型，且示意图如下所示。



<img src="picture/tongdao.jpg" width="50%">

### 1.1 数据描述

​		本文所有的状态参数均可以从数值分析结果 [1-2] 中获取，物理场数据是定义在结构化网格的每个节点上。我们采用Latin超立方采样方法采集了6773个样本，并划分为训练集（80%，5418个样本）和验证集（10%，677个样本作为验证集对模型进行调优）、测试集（10%，677 个样本）。之所以选择如此多的样本，是因为在开始仿真时，我们难以断定恢复全部流场需要多少的数据量。

|           | 数据格式                                                     | 数据组成                                                     |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **输入 ** | 设计变量：变量尺寸 10<br/>空间坐标：通道数 2，空间尺寸                792×40 | 设计变量：包括五个可变几何变量、一个表征纳米流体的工质特性的参数、两个表征边界条件的工况参数；此外，还包括两个常数，分别为入口温度 *T*= 293K 和出口压强*P* =100KPa 。<br/>空间坐标：包含三个方向，即x, y和z。注意，本文只取x和y两个方向。 |
| **输出**  | 物理场：通道数4 ，空间尺寸792×40     <br/>性能参数：空间尺寸 2 | 4个物理场：压强场p，温度场t，x方向的速度场u，y方向的速度场v。<br/>2个性能参数：Nusselt数和Fanning摩擦因子 |

​		微通道的*x*, *y*方向的尺寸分别为[0, 05] *μ*m和 [0, *a*] *μ*m，*a* = [130, 270] *μ*m。8个设计变量的取值如下所示。

| 设计变量 | *R*e | *φ*/% | *l*3/μm | *R*1/*μ*m | *R*2/*μ*m | *δ*1 | *δ*2 | *q*/W·m2 |
| :------: | :--: | :---: | :-----: | :-------: | :-------: | :--: | :--: | :------: |
| **下限** |  43  |  0.1  |   30    |    10     |    10     | -15  | -15  |  10,000  |
| **上限** | 1000 |  10   |   150   |    70     |    70     |  28  |  28  | 100,000  |



### 1.2 数据采样

​		建立HeatDataset类，继承于 'Dataset' 类，以便进行数据加载、预处理等。在初始化方法中，mode可取0，1，2，分别代表训练、验证和测试。'sample_size' , 'training_size' 和 'test_size' 分别表示选取样本数量，训练集占样本数量的比例和测试集占样本数量的比例。

```python
class HeatDataset(Dataset):
    TRAIN = 0
    VALID = 1
    TEST = 2
    def __init__(self, file, mode=0, sampler={'sample_mode': 'all', 'sample_size': 0.6},
                 shuffle=True, training_size=0.8, test_size=0.1):
```

### 1.3 数据读取

​		函数data_read实现了数据读取的功能，通过MatLoader加载我们的mat数据文件。首先，取mat文件内的 'data' 的第一维（即batchsize）定义为length。将读取到的 'data' 字段的数据按照 self.shuffle_idx 的顺序进行重新排列，然后将结果赋给了 self.data.design。 ' grids' 和 ' field'分别为空间坐标和温度场，在读取数据时，我们均在x方向将分辨率降低为原始数据的1倍，且 坐标只取前两个方向x和y。'target' 指性能参数，是 Nu 和 f 在最后一个维度上拼接起来的。

```python
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
        
        #注意原始数据在x方向分辨率降低了1倍
        self.data.coords = reader.read_field('grids')[:, ::2, :, :2][self.shuffle_idx]  
        self.data.fields = reader.read_field('field')[:, ::2, :, :][self.shuffle_idx]
        self.data.target = np.concatenate((reader.read_field('Nu'), reader.read_field('f')), axis=-1)  	 
        [self.shuffle_idx]
```



### 1.4 数据集划分

​		函数data_split实现了数据集划分的功能。训练集大小= 样本大小*训练集比例，验证集大小=max(样本大小 -（1-训练集大小 - 测试集大小）, 200)， 测试集大小=验样本大小-训练集大小-验证集大小。

​		当mode=0时，训练集中设计变量，坐标，物理场，性能参数均取从0到列表的第self.train_len个数据；当mode=1时，验证集中设计变量，坐标，物理场，性能参数均取从训练集d的第self.train_len个数据到训练集加验证集大小个数据；当mode=2时，测试集中设计变量，坐标，物理场，性能参数均取从从列表的倒数第 self.test_len 个元素开始，一直到列表的最后一个元素。

```python
    def data_split(self):

        self.train_len = int(self.data.length * self.training_size)                     #训练集长度
        self.valid_len = max(int(self.data.length * (1 - self.training_size - self.test_size)), 200)  #                                                    验证集长度
        self.test_len = self.data.length - self.train_len - self.valid_len         #测试集长度

        if self.mode == self.TRAIN:        #mode=0
            self.data.design = self.data.design[:self.train_len]
            self.data.coords = self.data.coords[:self.train_len]
            self.data.fields = self.data.fields[:self.train_len]
            self.data.target = self.data.target[:self.train_len]
            self.data.length = self.train_len
        elif self.mode == self.VALID:      #mode=1
            self.data.design = self.data.design[self.train_len:self.train_len + self.valid_len]
            self.data.coords = self.data.coords[self.train_len:self.train_len + self.valid_len]
            self.data.fields = self.data.fields[self.train_len:self.train_len + self.valid_len]
            self.data.target = self.data.target[self.train_len:self.train_len + self.valid_len]
            self.data.length = self.valid_len
        elif self.mode == self.TEST:       #mode=2
            self.data.design = self.data.design[-self.test_len:]
            self.data.coords = self.data.coords[-self.test_len:]
            self.data.fields = self.data.fields[-self.test_len:]
            self.data.target = self.data.target[-self.test_len:]
            self.data.length = self.test_len

```

### 1.5 数据加载

​		建立 'HeatDataLoader' 类，继承自  DataLoader 类，用于批量加载数据集。在初始化方法中，'batchsize' 为每个批次放入样本数，'shuffle' 为是否在每个训练时打乱样本的顺序，'drop_last' 为是否丢弃最后一个不满足一个样本数量的批次。

```python
class HeatDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False,):
        super(HeatDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                             drop_last=drop_last)
```

### 1.6 数据归一化

​		为了消除设计变量、空间坐标、物理场和性能参数之间的量纲差距，我们对数据进行了归一化操作。在代码中，提供了两种归一化方法，分别为 'min-max' 归一化和 'mean-std' 归一化。在本研究中，由于物理场的最大值和最小值具有明显的物理意义，因此我们采用 'min-max' 归一化方式。

```python
class DataNormer(object):
    """
        在最后一个维度进行归一化
    """
    def __init__(self, data, method="min-max", axis=None):
   
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


    def norm(self, x):           #数据归一化
        """
            输入张量
            参数 x: 输入张量
            返回 x: 输出张量
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

    def back(self, x):          #数据反归一化
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
```



## 3. 网络训练及验证

### 3.1 训练过程

​		定义train_epoch函数，旨在对网络进行训练。网络的输入为设计变量和空间坐标，物理场损失选择loss_func，即yaml文件 'basic_config' 的loss_name对应的损失类型。本研究中，我们选择MSE损失作为物理场总损失。

```python
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
```

### 3.2 网络验证

​		定义valid_epoch函数，旨在对网络进行验证。除了对物理场的求解，我们还添加了对性能参数的求解，即输入物理场、空间坐标和设计变量。物理场总损失和性能参数总损失均采用MSE损失，而fields_metric和target_metric分别表示四个物理场和两个性能参数的各自损失，具体计算方式由PhysicsLpLoss类通过求解相对范数来完成计算。其中，fields_metric采用二阶范数，target_metric采用一阶范数，同时考虑到性能参数f有接近于0的情况，因此使用relative=False。

```python
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
```

### 	3.3 网络推理测试

​		定义infer函数，旨在使用数据加载器 data_loader来获取对应的训练、验证或测试数据，同时使用 data_name 来生成对应文件。此部分，我们将所有训练数据、验证数据和测试数据分别进行求解，得到各自的物理场、性能参数的真实值、预测值以及测试值。同时，可视化二维物理场云图和性能参数回归图。

```python
 def infer(self, data_loader, data_name, show_nums=20):
```



### 	3.4 训练过程

​		定义train函数，旨在利用 train_loader 和 valid_loader 来获取训练数据和验证数据，将epoch、训练时间、验证时间、物理场总损失、性能参数总损失、4个物理场和2个性能参数各自的损失全部保存在loghistory.pkl 文件内，并把模型和训练的参数，例如epoch、数据配置、优化器、学习率、网络参数等信息全部保存在last_model.pdparams文件中。同时，对损失函数进行可视化。

```python
  def train(self, train_loader, valid_loader):
```



### 3.5 性能参数积分求解

​		定义 Characteristic 类，旨在利用神经算子网络预测的物理场，通过积分求解的方式，实现对性能参数 Nu 和 f 的预测。函数get_parameters_of_nano 实现了对纳米流体的物理性质（热导率、比热、密度、动力粘度）进行求解；函数 cal_f 、cal_tb 、cal_tw 实现了对控制方程中偏导数的求解。

```python
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
```



## 4. 网络架构及训练方法

### 4.1 FNO

#### 4.1.1 简介

FNO: Fourier Neural Operator

Reference: [3] Fourier Neural Operator for Parametric Partial Differential Equations

Github: https://github.com/zongyi-li/fourier_neural_operator

<img src="C:\Users\zt\Desktop\3.jpg" style="zoom:50%;" />

#### 4.1.2 代码说明

​		此神经算子网络的网络结构，如上图所示，主要由全连接层和傅里叶层组成。首先，采用设计变量和空间坐标作为输入，通过全连接层将输入通道从4提升到高维通道32。其次，通过4个傅里叶层（每层包括两个不同的操作，即顶部操作和底部操作），得到傅里叶层的输出。再通过两个全连接层，实现空间维度从398×42到396×40的变化，且通道数从42 先增加到128，并最终再降维回到目标维度4。

```python
class FNO2d(nn.Layer):
    """
        2维FNO网络
    """

    def __init__(self, in_dim, out_dim, modes=(8, 8), width=32, depth=4, steps=1, padding=2,                     activation='gelu', dropout=0.0):
        super(FNO2d, self).__init__()

        """
        1.通过全连接层 self.fc0 将输入通道提升到想要的维度.
        2. 4层积分算子 u' = (W + K)(u).
            W 由 self.w定义; K由self.conv 定义.
        3.  通过 self.fc1 和 self.fc2将空间维度降低回目标维度.

        输入:  10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        输入形状: (batchsize, x, y, c)
        输出: 下一个 timestep 的解
        输出形状: (batchsize, x, y, c)
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes = modes
        self.width = width
        self.depth = depth
        self.steps = steps
        self.padding = padding            # 如果输入是非周期性的，则进行填充
        self.activation = activation
        self.dropout = dropout
        self.fc0 = nn.Linear(steps * in_dim + 2, self.width)

        self.convs = nn.LayerList()
        for i in range(self.depth):
            self.convs.append(
                SpectralConv2d(self.width, self.width, self.modes, activation=self.activation,                   dropout=self.dropout))

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x, grid):
        """
        前向计算
        """
        if len(x.shape) != len(grid.shape):
            repeat_times = paddle.to_tensor([1]+grid.shape[1:-1]+[1], dtype='int32')
            x = paddle.tile(x[:, None, None, :], repeat_times=repeat_times)

        x = paddle.concat((x, grid), axis=-1)
        x = self.fc0(x)
        x = x.transpose((0, 3, 1, 2))

        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])  # 如果输入是非周期性的，则进行填充

        for i in range(self.depth):
            x = self.convs[i](x)

        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]
        x = x.transpose((0, 2, 3, 1))  # 如果输入是非周期性的，则进行填充
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
```

#### 4.1.3 参数配置

​		下面我们列出了FNO的具体训练参数和网络参数设置。其中，为了更清晰地展示参数配置，我们在'basic_config' 中列出的训练参数，均为针对每个特定模型训练时设置的，除此外剩余的基本训练参数设置，见 1.1 中的 'basic_config' 。

```python
basic_config:
  learning_rate: 1.e-3
  weight_decay: 1.e-9
  learning_beta:
    - 0.99
    - 0.99
  learning_milestones:
    - 200
    - 300
    - 400
  learning_gamma: 0.1

FNO_model:
  in_dim: 10         #输入维度
  out_dim: 4         #输出维度
  modes: 12          #模态
  width: 32          #网络宽度
  depth: 4           #网络深度
  steps: 1          
  padding: 2         #填充
  activation: 'gelu' #激活函数
  dropout: 0.0       #正则化方法，减少过拟合
```



### 4.2 U-Net

#### 4.1.1 简介

U-Net: 是一种卷积神经网络（CNN）方法，用于图像分割任务的深度学习模型，最初由Olaf Ronneberger等人在2015年提出。它的名字来源于其U形状的网络结构。

Reference: [4] U-Net: Convolutional Networks for Biomedical Image Segmentation

<img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014094749732.png" style="zoom: 50%;" />



#### 4.1.2 代码说明

​		在本网络架构中，我们设置网络宽度为32，深度为6，采用GELU激活函数。*E*n主要由一个2×2的最大池化层和2个3×3的卷积层组成。首先将输入图像进行上采样，通过插值改变张量尺寸，将空间尺寸从396×40减小为256×40，再经过2个3×3的卷积层。其次，我们进行2×2的下采样，再进行2个3×3的卷积层，保证空间尺寸不变化，如此循环5次后，通道数从最初的32变为了1024，图像空间尺寸从396×40变为8×2。同理，*B*o结构与*E*n相似，但只进行一次最大池化和两次卷积，此时，通道数变为2048，空间尺寸变为4×1。*D*e主要由一个2×2的转置卷积，一个特征融合，两个3×3的卷积层组成。如此6次循环后，通道数从1024变回为32，空间尺寸从8×2变回为256×64。最后，我们经过一个上采样，将空间尺寸调整为396×40，通道数减少为4，最后再经过一个3×3的卷积层后，输出我们想要的4个物理场。

```python
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
```

#### 4.1.3 参数配置

​	下面我们列出了U-Net的具体训练参数和网络参数设置。其中，为了更清晰地展示参数配置，我们在'basic_config' 中列出的训练参数，均为针对每个特定模型训练时设置的，除此外剩余的基本训练参数设置，见 1.1 中的 'basic_config' 。

```python
basic_config:
  total_epoch: 500
  learning_rate: 1.e-4
  weight_decay: 1.e-12
  learning_beta:
    - 0.7
    - 0.9
  learning_milestones:
    - 300
    - 400
    - 500
  learning_gamma: 0.1

CNN_model:
  in_sizes: [396, 40, 10]
  out_sizes: [396, 40, 4]
  width: 32
  depth: 6
  steps: 1    # Number of steps to unroll, should be 1 in this problem
  activation: 'gelu'
  dropout: 0.0
```



### 4.3 FNN

#### 4.1.1 简介

FNN: Fully Neural Network

Reference: [5] A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics

<img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014093121334.png" style="zoom: 50%;" />

#### 4.1.2 代码说明

​		在本研究中，我们采用多个全连接层，每个全连接层来预测一个物理场。输入维度为设计变量+空间坐标=10，网络宽度为64，深度为5，输出维度是4，即压强场、温度场、速度场*u*和*v*。 

```python
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
```

#### 4.1.3 参数配置

​	下面我们列出了FNN的具体训练参数和网络参数设置。其中，为了更清晰地展示参数配置，我们在'basic_config' 中列出的训练参数，均为针对每个特定模型训练时设置的。除此外剩余的基本训练参数设置，见 1.1 中的 'basic_config' 。

```python
basic_config:
 
  total_epoch: 800
  learning_rate: 1.e-3
  weight_decay: 0.
  learning_beta:
    - 0.7
    - 0.9
  learning_milestones:
    - 600
    - 700
    - 800
  learning_gamma: 0.1


MLP_model:
  in_dim: 10
  out_dim: 4
  steps: 1
  planes:
    - 64
    - 64
    - 64
    - 64
    - 64
  activation: 'gelu'
```



### 4.4 DeepONet

#### 4.1.1 简介

DeepONet: deep operator network

Reference: [6] Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators

<img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014093758871.png" style="zoom: 80%;" />

#### 4.1.2 代码说明

​		DeepONet是深度神经算子网络，由主干网络Trunk net和分支网络Branch net组成，通过最小化目标算子与给定的神经网络之间的误差来实现对复杂特征的准确预测。其中，该网络支持搭建多个Branch net，例如边界条件是分布式的输入。在对在对该网络进行训练时，我们Branch net的输入维度为包含10个设计变量的列表，Trunk net的输入维度为包含2个空间坐标的张量，分别设置二者的隐藏层数为[64, 64, 64, 64, 64]和 [64, 64, 64, 64]。

```python
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
```



#### 4.1.3 参数配置

​	下面我们列出了DeepONet的具体训练参数和网络参数设置。其中，为了更清晰地展示参数配置，我们在'basic_config' 中列出的训练参数，均为针对每个特定模型训练时设置的，除此外剩余的基本训练参数设置，见 1.1 中的 'basic_config' 。

```python
basic_config:
  total_epoch: 800
  learning_rate: 1.e-2
  weight_decay: 0.
  learning_beta:
    - 0.7
    - 0.9
  learning_milestones:
    - 600
    - 700
    - 800
  learning_gamma: 0.1


DON_model:
  in_dim: 2                # note: deeponet in_dim 为空间坐标
  out_dim: 4
  operator_dims:           # note: deeponet operator_dims 为设计变量
    - 10
  planes_branch:
    - 64
    - 64
    - 64
    - 64
    - 64
  planes_trunk:
    - 64
    - 64
    - 64
    - 64
  activation: 'gelu'
```



### 4.5 Transformer

#### 4.1.1 简介

reference：[7] Choose a Transformer: Fourier or Galerkin

github：https://github.com/scaomath/galerkin-transformer

| 形式     | Attention计算方式                                            |                                                              |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Fourier  | <img src="C:\Users\zt\Desktop\or2\a62efc6b06d3441492c507b1c381521fe474fcf9eed94ee6b53a7c36cd24184b.jpg" style="zoom: 50%;" /> | <img src="C:\Users\zt\Desktop\or2\2.jpg" style="zoom: 33%;" /> |
| Galerkin | <img src="C:\Users\zt\Desktop\or2\3.jpg" style="zoom: 50%;" /> | <img src="C:\Users\zt\Desktop\or2\4.jpg" style="zoom: 33%;" /> |

#### 4.1.2 代码说明

​		改进的attention机制核心算法实现过程：（包括fourier,galerkin,linear,softmax）

```python
class SimpleAttention(nn.Layer):
    '''
    The attention is using a vanilla (QK^T)V or Q(K^T V) with no softmax
    For an encoder layer, the tensor size is slighly different from the official pytorch implementation

    attn_types:
        - fourier: integral, local
        - galerkin: global
        - linear: standard linearization
        - softmax: classic softmax attention

    In this implementation, output is (N, L, E).
    batch_first will be added in the next version of PyTorch: https://github.com/pytorch/pytorch/pull/55285

    Reference: code base modified from
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    - added xavier init gain
    - added layer norm <-> attn norm switch
    - added diagonal init

    In https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    the linear attention in each head is implemented as an Einstein sum
    attn_matrix = paddle.einsum('bhnd,bhne->bhde', k, v)
    attn = paddle.einsum('bhnd,bhde->bhne', q, attn_matrix)
    return attn.reshape(*q.shape)
    here in our implementation this is achieved by a slower transpose+matmul
    but can conform with the template Harvard NLP gave
    '''

    def __init__(self, n_head, d_model,
                 pos_dim: int = 1,
                 attention_type='fourier',
                 dropout=0.1,
                 xavier_init=1e-4,
                 diagonal_weight=1e-2,
                 symmetric_init=False,
                 norm_add=False,
                 norm_type='layer',
                 eps=1e-5):
        super(SimpleAttention, self).__init__()
        assert d_model % n_head == 0  # n_head 可被d_model整除
        self.attention_type = attention_type
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.linears = nn.LayerList(
            [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(3)])
        self.xavier_init = xavier_init
        self.diagonal_weight = diagonal_weight
        self.symmetric_init = symmetric_init
        if self.xavier_init > 0:
            self._reset_parameters()
        self.norm_add = norm_add
        self.norm_type = norm_type
        if norm_add:
            self._get_norm(eps=eps)

        if pos_dim > 0:
            self.fc = nn.Linear(d_model + n_head * pos_dim, d_model)

        self.attn_weight = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, pos=None, mask=None, weight=None):
        """
        forward compute
        :param query: (batch, seq_len, d_model)
        :param key: (batch, seq_len, d_model)
        :param value: (batch, seq_len, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)

        bsz = query.shape[0]
        if weight is not None:
            query, key = weight * query, weight * key

        query, key, value = \
            [layer(x).reshape((bsz, -1, self.n_head, self.d_k)).transpose((0, 2, 1, 3))
             for layer, x in zip(self.linears, (query, key, value))]

        if self.norm_add:
            if self.attention_type in ['linear', 'galerkin', 'global']:
                if self.norm_type == 'instance':
                    key, value = key.transpose((0, 1, 3, 2)), value.transpose((0, 1, 3, 2))

                key = paddle.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], axis=1)
                value = paddle.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_V, (value[:, i, ...] for i in range(self.n_head)))], axis=1)

                if self.norm_type == 'instance':
                    key, value = key.transpose((0, 1, 3, 2)), value.transpose((0, 1, 3, 2))
            else:
                if self.norm_type == 'instance':
                    key, query = key.transpose((0, 1, 3, 2)), query.transpose((0, 1, 3, 2))

                key = paddle.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], axis=1)
                query = paddle.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_Q, (query[:, i, ...] for i in range(self.n_head)))], axis=1)

                if self.norm_type == 'instance':
                    key, query = key.transpose((0, 1, 3, 2)), query.transpose((0, 1, 3, 2))

        if pos is not None and self.pos_dim > 0:
            assert pos.shape[-1] == self.pos_dim
            pos = pos.unsqueeze(1)
            pos = pos.tile([1, self.n_head, 1, 1])
            query, key, value = [paddle.concat([pos, x], axis=-1)
                                 for x in (query, key, value)]

        if self.attention_type in ['linear', 'galerkin', 'global']:
            x, self.attn_weight = linear_attention(query, key, value,
                                                   mask=mask,
                                                   attention_type=self.attention_type,
                                                   dropout=self.dropout)
        else:
            x, self.attn_weight = vanilla_attention(query, key, value,
                                                    mask=mask,
                                                    attention_type=self.attention_type,
                                                    dropout=self.dropout)

        out_dim = self.n_head * self.d_k if pos is None else self.n_head * \
                                                             (self.d_k + self.pos_dim)
        att_output = x.transpose((0, 2, 1, 3)).reshape((bsz, -1, out_dim))

        if pos is not None and self.pos_dim > 0:
            att_output = self.fc(att_output)

        return att_output, self.attn_weight
```

#### 4.1.3 参数配置

​		下面我们列出了Transformer的具体训练参数和网络参数设置。其中，为了更清晰地展示参数配置，我们在'basic_config' 中列出的训练参数，均为针对每个特定模型训练时设置的，除此外剩余的基本训练参数设置，见 1.1 中的 'basic_config' 。

```python
basic_config:
  total_epoch: 800
  learning_rate: 1.e-3
  weight_decay: 0.
  learning_beta:
    - 0.9
    - 0.99
  learning_milestones:
    - 600
    - 700
    - 800
  learning_gamma: 0.1

TNO_model:
  node_feats: 10    #设计变量
  pos_dim: 2        #空间坐标
  n_targets: 4      #目标变量
  n_hidden: 64
  num_feat_layers: 0   #特征提取曾数目
  num_encoder_layers: 4  #编码层数目
  n_head: 4
  normalizer: False
  dim_feedforward: 128
  residual_type: add
  attention_type: galerkin
  attn_activation: gelu
  feat_extract_type: None
  xavier_init: 0.01
  diagonal_weight: 0.01
  symmetric_init: False
  layer_norm: True
  attn_norm: False
  norm_eps: 0.00001
  batch_norm: False
  return_attn_weight: False
  return_latent: False
  decoder_type: ifft2
  last_activation: True
  freq_dim: 32
  num_regressor_layers: 2
  regressor_activation: gelu
  fourier_modes: 16
  spacial_dim: 2    #空间维度
  spacial_fc: False  #是否添加空间维度到输入维度
  dropout: 0.0
  encoder_dropout: 0.0
  decoder_dropout: 0.0
  ffn_dropout: 0.0
```



### 4.6 训练方法

​		五种神经算子网络均采用相同的训练方式，即Adam优化器。FNO、U-Net、FNN、DeepONet和Transformer的训练步长分别采用400，500，800，800和800个Epoch。损失函数采用三种方式：训练总损失和性能参数损失采用*L*2（MSE）损失；物理场损失采用平均相对误差和最大相对误差。损失函数具体公式如下所示：

| 损失函数                   | 公式                                                         |
| -------------------------- | ------------------------------------------------------------ |
| **物理场及性能参数总损失** | <img src="C:\Users\zt\Desktop\L2.png" style="zoom: 50%;" />  |
| **物理场  ：平均相对误差** | <img src="C:\Users\zt\Desktop\MAX.png" style="zoom: 50%;" /> |
| **物理场  ：最大相对误差** | <img src="C:\Users\zt\Desktop\MEA.png" style="zoom:50%;" />  |
| **性能参数：相对误差**     | <img src="C:\Users\zt\Desktop\canshu.png" style="zoom:50%;" /> |



## 5. 结果分析

- 下述所有结果仅展示了部分分析，详细的结论请参见文章中。

​		下图展示了五种神经算子网络模型在整个训练过程中的总物理场损失、每个物理场的损失及性能参数收敛情况。

| 损失收敛曲线                 |                                                              |
| ---------------------------- | ------------------------------------------------------------ |
| **物理场总损失**             | ![](C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014084848857.png) |
| **性能参数总损失**           | <img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014084933092.png" style="zoom:23%;" /> |
| **四个物理场的平均相对误差** | <img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014085044410.png" style="zoom:23%;" /> |
| **四个物理场的最大相对误差** | <img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014085134461.png" style="zoom:23%;" /> |

​		为了更直观地比较不同模型在预测物理场中的偏差，我们选择60%的样本量，对五种神经算子网络模型预测物理场的最大相对误差和平均相对误差进行对比。

| FMAXD     | <img src="C:\Users\zt\Desktop\image-20231014073800647.png"  /> |
| --------- | ------------------------------------------------------------ |
| **FMEAD** | ![](C:\Users\zt\Desktop\image-20231014073818625.png)         |

​		下图展示了五种神经算子网络预测的压力场p、温度场t、速度场u和v的真实场、预测场和误差分布。

| 预测全局物理场  |                                                            |
| --------------- | ---------------------------------------------------------- |
| **FNO**         | <img src="C:\Users\zt\Desktop\1.jpg" style="zoom: 50%;" /> |
| **U-Net**       | <img src="C:\Users\zt\Desktop\2.jpg" style="zoom:50%;" />  |
| **FNN**         | <img src="C:\Users\zt\Desktop\4.jpg" style="zoom:50%;" />  |
| **DeepONet**    | <img src="C:\Users\zt\Desktop\5.jpg" style="zoom:50%;" />  |
| **Transformer** | <img src="C:\Users\zt\Desktop\7.jpg" style="zoom:50%;" />  |

​		此处，我们只放置预测效果最好的Tranformer和最差的DeepONet的物理场局部放大图。

| 预测局部物理场  |                                                             |
| --------------- | ----------------------------------------------------------- |
| **Transformer** | <img src="C:\Users\zt\Desktop\11.jpg" style="zoom: 33%;" /> |
| **DeepONet**    | <img src="C:\Users\zt\Desktop\22.jpg" style="zoom:33%;" />  |

​			为了更直观地可视化微通道内每个物理变量的变化趋势，并对比不同神经算子网络模型在捕捉局部物理场细节的能力，我们选择了三个特定的位置进行分析，分析方法受文章[8]启发。这些位置分别位于靠近微通道的上壁、下壁和中间区域。曲线U位于距离上壁6.30 *μ*m处，曲线D位于距离下壁6.30 *μ*m处，曲线M位于微通道中心，距离上下壁面等距2.5 mm。注意：为了更清楚地反映三个不同通道位置的物理变量变化趋势及不同神经算子网络模型的局部预测效果，我们在此处只任选两种神经算子网络FNO和U-Net与真实值进行对比。其中蓝线为真实值，红圈、黑圈分别表示FNO和U-Net的预测值。值得注意的是，两种神经算子网络模型预测的物理变量与实际值基本一致，但在非矩形区域的凹槽和凸槽附近，U-Net很难捕捉到局部细微特征，因此预测误差较大。	

| 微通道内部不同位置的物理量分析 | 曲线D                                                        | 曲线M                                                        | 曲线U                                                        |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **压强p**                      | <img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014075733456.png" style="zoom: 20%;" /> | <img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014080010767.png" style="zoom:25%;" /> | <img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014080149573.png" style="zoom:20%;" /> |
| **温度t**                      | <img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014080331865.png" style="zoom:30%;" /> | <img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014080452411.png" style="zoom:30%;" /> | <img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014080547025.png" style="zoom:25%;" /> |
| **速度u**                      | <img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014080705280.png" style="zoom:25%;" /> | <img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014080809356.png" style="zoom:25%;" /> | <img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014080850810.png" style="zoom:25%;" /> |
| **速度v**                      | <img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014080930512.png" style="zoom:25%;" /> | <img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014081017587.png" style="zoom:25%;" /> | <img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014081147769.png" style="zoom:25%;" /> |

​		在本文中，我们成功地预测了流体的物理场，并以空间积分的形式提取了有效表征流体传热性能的热物性参数*Nu*和*f*。如下图所示，*x*轴表示实际值，*y*轴表示预测值。散点越接近*y* *= x*线，预测结果越准确。值得注意的是，*Nu* 和 *f* 的预测误差分别在6%和5%以内，这证实了我们的方法可以较高地精度地实现对流动换热性能参数地预测。

| 性能参数 |                                                              |
| -------- | ------------------------------------------------------------ |
| ***Nu*** | <img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014083121521.png" style="zoom:25%;" /> |
| ***f***  | <img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014083147236.png" style="zoom:25%;" /> |

​		为了讨论不同样本数量对预测效果的影响，我们任选两种预测效果较好的神经算子网络FNO和效果较差的U-Net，分别选取总样本数量的60%、40%、20%、10%、5%和2.5%，对预测效果进行对比。综合FNO和U-Net在不同样本量下物理场的平均绝对预测误差，我们可以看出，FNO在较少样本量下实现了较高精度的物理场预测。例如只选取5%的样本即可达到小于0.1的物理场误差，选取40%的样本可以获得最佳的预测效果。而U-Net需要40%的样本才可达到小于0.1的误差，60%的样本达到最优的预测效果。因此，在训练样本数量的角度来看，FNO的计算所需资源远小于U-Net。

| 训练集大小对物理场平均相对误差的影响 | FNO                                                          | U-Net                                                        |
| ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **FMAEAD**                           | <img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014083623926.png" style="zoom:25%;" /> | <img src="C:\Users\zt\AppData\Roaming\Typora\typora-user-images\image-20231014083721214.png" style="zoom:25%;" /> |

​		最后，我们在计算成本方面对五种神经算子网络模型进行了更详细的比较。如下表所示，FNO占用显存最少，DeepONet占用显存最多，且分别需要占用1.21 GB和 6.43 GB的显存，后者的需求大约是前者的5倍。在训练时间方面，DeepONet 训练时间最长，相比时间最短的FNO增加了3 倍。此外，Transformer和U-Net的总参数量最多，大量数据需求是由于其大量的内存成本，较长的训练时间，以及需要大量的训练样本来实现对物理场的精确识别。但综合物理场及损失等对比，Transformer和FNO在预测精度和训练成本上最优，U-Net和DeepONet的预测精度和训练成本最差。且预测效果最优的Transformer比预测效果最差的DeepONet的预测精度提高了1个数量级。

| 训练成本            | FNO       | U-Net       | FNN     | DeepONet | Transformer |
| ------------------- | --------- | ----------- | ------- | -------- | ----------- |
| 步长                | 400       | 500         | 800     | 800      | 800         |
| 训练样本数/总样本数 | 0.6       | 0.6         | 0.6     | 0.6      | 0.6         |
| 总参数量            | 2,106,468 | 212,488,324 | 120,263 | 6,8032   | 21,117,220  |
| 模型大小 (MB)       | 0.04      | 547.99      | 0.28    | 0.19     | 54.11       |
| 显存占用 (GB)       | 1.21      | 2.17        | 5.62    | 6.43     | 2.48        |
| 训练时间 (h)        | 1.02      | 1.81        | 3.11    | 3.25     | 1.70        |

​		**本次代码和测试结果仅仅抛砖引玉，期待有兴趣的使用者在此基础上进行进一步探索。**



## 6.模型信息

| 信息     | 说明                              |
| -------- | --------------------------------- |
| 发布者   | tianshao1992                      |
| 时间     | 2023.10                           |
| 框架版本 | Paddle Develope                   |
| 应用场景 | 科学计算                          |
| 支持硬件 | AMD EPYC 7642 CPU、Nvidia A40 GPU |



## Reference

[1]   LIU T Y, LI Y Z, XIE Y H, et al. Deep Learning for Nanofluid Field Reconstruction in Experimental Analysis [J]. IEEE Access, 2020, 8: 64692-706.

[2]  LIU T Y, LI Y Z, JING Q, et al. Supervised learning method for the physical field reconstruction in a nanofluid heat transfer problem [J]. International Journal of Heat and Mass Transfer, 2021, 165: 24.

[3]  Li Z Y,  Zheng N, NIKOLA Kovachki, et al., Physics-Informed Neural Operator for Learning  Partial Differential Equations [J], arXiv: 2111.03794, 2021. 

[4] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. In: Navab N, Hornegger J, Wells WM, Frangi AF, editors. Med. Image Comput. Comput. Interv. – MICCAI 2015, Cham: Springer International Publishing; 2015, p. 234–41.

[5] HAGHIGHAT E, RAISSI M, MOURE A, et al. A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics [J]. Comput Meth Appl Mech Eng, 2021, 379: 22.

[6] LU L, JIN P Z, PANG G F, et al. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators [J]. Nat Mach Intell, 2021, 3(3): 218.

[7] CAO S H. Choose a Transformer: Fourier or Galerkin; proceedings of the Neural Information Processing Systems, F, 2021 [J]. arXiv: 2105.14995, 2021.

[8] LI Y Z, LIU T Y, XIE Y H. Thermal fluid fields reconstruction for nanofluids convection based on physics-informed deep learning [J]. Sci Rep, 2022, 12(1): 23.

