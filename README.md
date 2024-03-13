# MyTorch
本项目参考pytorch，needle等深度学习库的设计方式和源代码，编写了一个用动态图来计算的深度学习框架。
![logo](./figures/logo.png)
## **目录**

- **[项目背景](#项目背景)**
- **[项目组成](#项目组成)**
- **[安装步骤](#安装步骤)**
- **[快速开始](#快速开始)**
- **[贡献列表](#贡献列表)**
 

## 项目背景
   

### 动机
某人在暑假学习了如何搭建深度学习框架，在秋季迫于课程的安排，召集小伙伴们，耗时约三周做出了此框架

### 相关背景知识

#### 1. 计算图原理
计算图是描述数学运算和数据流的一种方式。在深度学习中，它以图的形式表示数据在神经网络中的流动。节点代表操作，边表示数据流。了解计算图的原理有助于理解神经网络的计算过程和优化方法，如反向传播。[Mytorch中的计算图](./docs/AutoGrad.md)

#### 2. 模型网络模块
深度学习模型通常由多个网络模块组成，比如：
- **卷积层（Convolutional Layers）**: 用于提取输入数据中的空间信息。
- **全连接层（Fully Connected Layers）**: 执行分类任务和输出预测结果。
每个模块都有不同的功能和作用，构成了深度学习模型的基本构建块。
详情见： [Mytorch提供的网络模块](./docs/NeuralNetworks.md)

#### 3. 优化器方法
优化器是用于训练神经网络的算法，它们通过调整模型的参数以最小化损失函数。常见的优化器方法包括：
- **随机梯度下降（Stochastic Gradient Descent，SGD）**
- **Adam** [有关adam的一切](https://arxiv.org/pdf/1412.6980.pdf)

#### 4. 数据加载
数据加载是指将数据集导入到机器学习模型中进行训练和测试的过程。在深度学习中，数据加载通常包括数据预处理、批处理和输入模型等步骤。了解有效的数据加载方法有助于更高效地训练模型和处理数据。[Mytorch中的数据加载](./docs/Data.md)



## 项目组成
展示核心组件
```
├─README.md
├─python
|   ├─test
|   ├─mytorch（项目根目录）
|   |    ├─array_device.py（设备后端选择）
|   |    ├─autograd.py （自动求导）
|   |    ├─array_api.py （数组后端选择）
|   |    ├─optim.py （优化器）
|   |    ├─perfomance.py （性能展现）
|   |    ├─__init__.py
|   |    ├─ops
|   |    |  ├─ops.py （算子计算与反传）
|   |    |  ├─__init__.py
|   |    ├─nn
|   |    | ├─nn.py （网络模块）
|   |    | ├─__init__.py
|   |    ├─init
|   |    |  ├─initializer.py （初始化方法）
|   |    |  ├─__init__.py
|   |    ├─data
|   |    |  ├─data_basic.py （数据加载）
|   |    |  ├─data_transforms.py （数据裁剪）
|   |    |  ├─__init__.py
|   |    |  ├─datasets
|   |    |  |    ├─mnist_dataset.py （MNIST数据集）
|   |    |  |    ├─__init__.py
├─figures
├─docs
├─data
```

## 安装步骤

本项目有两种使用方式：

### 1. 使用方式一：

在 `python/test` 目录下仿照测试文件的引用方法，在 `sys.path` 中加入 `python` 文件夹的目录。即可开始使用。
```python
import sys
import os

# 获取当前工作目录
current_dir = os.getcwd()

# 获取父目录（即 python_dir）
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)


import numpy as np
import cupy as cp
import mytorch as torch
import mytorch.ops as ops
import mytorch.nn as nn
import mytorch.optim as optim
import mytorch.data as data
from mytorch.array_device import *
from mytorch.array_api import array_api
import mytorch.perfomance as pf
```

**注意：这种方法是一次性的，推荐给测试开发人员使用**

### 2. 使用方式二：

在 `python/mytorch` 目录下打开命令行：

**注意：请确保你已经安装了 Conda 和 CUDA 11.x 以上的版本。如果你的电脑没有 NVIDIA 的显卡，无法使用本库。**

- 创建一个 Python 虚拟环境：

```bash
conda create --name Mytorch python=3.10
```

- 然后手动安装CuPy和NumPy （CuPy根据你的CUDA版本决定）
如果你不确定，可以查看 [CuPy官网](https://docs.cupy.dev/en/stable/install.html)
比如我的CUDA是11.8版本,那么在刚刚创建的环境下输入命令
```bash
pip install numpy
pip install cupy-cuda11x
```

- 在mytorch目录下安装Mytorch
```bash
pip install .
```

## 快速开始

为了方便用户快速上手我们的项目，这里给出一些简单的示例
- 第一步导入mytorch（确保经历了安装项目这一步）
```python
import mytorch as torch
```
- 创建一个简单的tensor看看
```python
x = torch.Tensor([1,2,3]) #默认创建在cpu端口
print(x)
```
如果没有问题会显示`mytorch.Tensor([1,2,3])`

- 创建gpu端的tensor
```python
y = torch.Tensor([1,2,3],device = gpu()) 
print(y.device)
```
如果没有问题会显示`mytorch.gpu()`

- 转移cpu的tensor到gpu
```python
x.to('gpu')
print(x.device)
```
如果没有问题会显示`mytorch.gpu()`

- 定义一个简单的全连接网络（这里指定都在gpu端）
```python
# 导入必要的包
import numpy as np
import cupy as cp
import mytorch as torch
import mytorch.nn as nn
import mytorch.optim as optim
import mytorch.data as data
from mytorch.array_device import *
import mytorch.perfomance as pf

# 定义一个全连接网络
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,100,device=gpu()),
            nn.ReLU(),
            nn.Linear(100,10,device=gpu()),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)
```

- 设置超参数，优化器，损失函数
```python
batch_size=100
epochs=10
net = Model(device=gpu())
optimizer=optim.Adam(net.parameters(),lr=0.001,weight_decay=0.001)
criterion= nn.SoftmaxLoss()
```

- 加载数据集和使用数据装载器
```python
import os
# 获取根目录 这里读的数据是我们项目自带的MNIST数据集
root_dir= os.path.abspath(os.path.join(project_dir, os.pardir))
root_dir

# 加载训练数据集
train_dataset = data.MNISTDataset(\
        f"{root_dir}/data/MNIST/train-images-idx3-ubyte.gz",
        f"{root_dir}/data/MNIST/train-labels-idx1-ubyte.gz")

# 训练集装入装载器，方便打乱和批量计算
train_dataloader = data.DataLoader(\
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True)

# 测试集同理
test_dataset = data.MNISTDataset(\
        f"{root_dir}/data/MNIST/t10k-images-idx3-ubyte.gz",
        f"{root_dir}/data/MNIST/t10k-labels-idx1-ubyte.gz")
test_dataloader = data.DataLoader(\
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True)
```

- 开始训练
```python
# 创建一个性能展示对象，再也不用手动画图
pfm = pf.Performance()
# 迭代
for epoch in range(epochs):
    # 一些计算参数
    total_loss = 0
    total_rights = 0
    total_examples = 0
    total_batches = 0
    # 从装载器取出数据
    for input,label in train_dataloader:
        # 将数据放到gpu上
        input.to('gpu')
        label.to('gpu')
        # 将模型设置为训练模式
        # （实际上训练模式和评估模式的区别只在于部分网络层的计算规则）
        net.train()
        # 梯度清0
        optimizer.reset_grad()
        # 前向计算网络预测的答案
        pred = net(input)
        # 利用损失函数计算损失
        loss = criterion(pred,label)
        # 根据构建好的计算图反向传播
        loss.backward()
        # 利用优化器更新所有参数
        optimizer.step()
        # 取出预测的标签
        label_pred = cp.argmax(pred.cupy(),axis=1)
        # 计算出正确的标签个数
        rights = cp.equal(label_pred, label.cupy()).sum()

        # 每一个batch做一次更新
        total_loss += loss.cupy()
        total_rights += rights
        total_batches += 1
        total_examples += input.shape[0]
    
    # 计算平均损失
    avg_loss = total_loss / total_batches
    # 计算平均准确率
    avg_accuracy = total_rights / total_examples
    # 更新性能展示对象
    pfm.update_in_train(avg_accuracy,avg_loss)
    # 打印损失和准确率
    print(f"EPOCH {epoch}: {avg_accuracy=}, {avg_loss=}")
# 展示损失曲线
pfm.graph()
```

- 开始测试
```python
# 与训练类似，不再给出详细注释
total_loss = 0
total_rights = 0
total_examples = 0
total_batches = 0
for input,label in test_dataloader:
    input.to('gpu')
    label.to('gpu')
    net.eval()

    pred = net(input)
    loss = criterion(pred,label)
    label_pred = cp.argmax(pred.cupy(),axis=1)
    rights = cp.equal(label_pred, label.cupy()).sum()

    total_loss += loss.cupy()
    total_rights += rights
    total_batches += 1
    total_examples += input.shape[0]
    pfm.update_in_test(label.cupy(), label_pred)
    
avg_loss = total_loss / total_batches
avg_accuracy = total_rights / total_examples

print(f"TEST SCORE: {avg_accuracy=}, {avg_loss=}")

# 画出混淆矩阵
pfm.matrix(8)
```
如果过程没有问题，运气不算太差，那么理论能跑到97%的准确率，得益于参数初始化的`kaiming`分布, 以及softmax损失函数精度的优化。


## 贡献列表

#### 项目负责人
- MR (meiran0528@gmail.com)

#### 首席技术指导
- ZJY (zjyscu@gmail.com)

#### 创意策划
- LSX (1275425660@qq.com)
