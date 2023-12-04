# 神经网络模块

mytorch里提供了丰富的网路模块供炼丹师们训练，下面我们介绍一下这些模块的作用。

- **恒等映射函数（Identity）**
    - **参数：**
        - 无参数
    - **计算：**
        - 输出等于输入
    - **使用方法：**
        ```python
        from mytorch.nn import Identity
        
        # 示例使用参数解包函数
        identity_module = Identity()
        output = identity_module(input_tensor)
        ```

- **全连接层（Identity）**
    - **参数：**
        - `in_features`(int)：输入特征的数量。
        - `out_features`(int)：输出特征的数量。
        - `bias`(bool,可选)：是否包含偏置，默认为True。
        - `device`(str，可选)：张量所在的设备，例如“cuda”或“cpu”；默认为None。
        - `dtype`（str，可选）：张量的数据类型，例如“float32”。默认为“float32”。
    - **计算：**
        - 进行输入和权重的矩阵乘法，然后加上偏置项（如果启用）。
    - **使用方法：**
        ```python
        from mytorch.nn import Linear
        
        # 示例使用参数解包函数
        linear_module = Linear(in_features = 100,out_features=50)
        output = linear_module(input_tensor) 
        ```

- **Flatten模块**
    - **参数：**
        - 无

    - **计算：**
        - 将输入张量使用`ops.reshape`函数进行展平操作。
    - **使用方法：**
        ```python
        from mytorch.nn import Flatten

       # 示例创建Flatten模块
      flatten_module = Flatten()
      output = flatten_module(input_tensor)
      class Flatten(Module)：
        def forward(self, X):
            return ops.reshape(X, (X.shape[0], -1))
        ```

- **ReLu激活模块（ReLu）**
    - **参数：**
        - 无
    - **计算：**
        - 用`ops.reshape`函数对输入张量进行ReLu激活
    - **使用方法：**
        ```python
        from mytorch.nn import ReLU

      # 示例创建ReLU激活模块  
       relu_module = ReLU()
       output = relu_module(input_tensor)
        ```

- **序列模块（Sequential）**
    - **参数：**
        - `*modules`(Module):以顺序执行的模块列表
    - **计算：**
        - 依次对输入张量应用模块列表中的每个模块，得到最终的输出张量。
    - **使用方法：**
        ```python
        from mytorch.nn import Sequential, Flatten, ReLU

      # 示例创建序列模块
       seq_module = Sequential(
        Flatten(),
        ReLU()
        )
      output = seq_module(input_tensor)
        ```
- **交叉熵损失模块（Softmax）**
    - **参数：**
        - `logits`(Tensor):模型的原始输出，通常是未经过Softmax激活的预测值。
    - **计算：**
        - 计算Softmax函数的对数归一化，然后通过标签的独热编码进行损失计算。
    - **使用方法：**
        ```python
       from mytorch.nn import SoftmaxLoss

      # 示例创建SoftmaxLoss模块
      softmax_loss_module = SoftmaxLoss()
      loss = softmax_loss_module(logits, true_labels)
        ```
- **批归一化模块（BatchNorm1d）**
    - **参数：**
        - `dim`(int):特征维度。
        - `eps`(float,可选):避免除零错误的小值，默认为1e-5。
         - `momentum`(float,可选):动量参数，用于计算移动平均，默认为0.1。
        - `device`(str,可选):张量所在的设备，例如"cuda"或"cpu"；默认为None。
        - `dtype`(str,可选):张量的数据类型，例如"float32"。默认为"float32"。
    - **计算：**
        - 在训练时，计算均值、方差，更新移动平均；在测试时使用移动平均进行归一化。
    - **使用方法：**
        ```from mytorch.nn import BatchNorm1d

      # 示例创建BatchNorm1d模块
      batch_norm_module = BatchNorm1d(dim=64)
      normalized_output = batch_norm_module(input_tensor)
        ```
- **批归一化模块（BatchNorm2d）**
    - **参数：**
        - `BatchNorm1d`:无额外参数。
    - **计算：**
        - 在训练时，计算均值、方差，更新移动平均；在测试时使用移动平均进行归一化。
    - **使用方法：**
        ```python
        from mytorch.nn import BatchNorm2d

      # 示例创建BatchNorm2d模块
      batch_norm_2d_module = BatchNorm2d(dim=64)
      normalized_output = batch_norm_2d_module(input_tensor)
        ```
- **层归一化模块（Sequential）**
    - **参数：**
        - `dim`(int):特征维度。
        - `eps`(float,可选):避免除零错误的小值，默认为1e-5。
        - `device`(str,可选):张量所在的设备，例如"cuda"或"cpu"；默认为None。
        - `dtype`(str,可选):张量的数据类型，例如"float32"。默认为"float32"。
    - **计算：**
        - 计算均值、方差，然后通过归一化公式进行操作。
    - **使用方法：**
        ```python
        from mytorch.nn import LayerNorm1d

      # 示例创建LayerNorm1d模块
      layer_norm_module = LayerNorm1d(dim=64)
      normalized_output = layer_norm_module(input_tensor)
        ```
- **Dropout模块（Dropout）**
    - **参数：**
        - ` p`(float,可选):丢弃概率，默认为0.5。
    - **计算：**
        - 在训练时，随机生成掩码并应用到输入张量上。
    - **使用方法：**
        ```python
        from mytorch.nn import Dropout

      # 示例创建Dropout模块
      dropout_module = Dropout(p=0.5)
      output = dropout_module(input_tensor)
        ```
- **残差连接模块（Residual）**
    - **参数：**
        - `fn`(Module):用于处理输入的子模块。
    - **计算：**
        - 将输入张量与子模块处理后的张量相加
    - **使用方法：**
        ```python
        from mytorch.nn import Residual, SomeModule

      # 示例创建Residual模块
      residual_module = Residual(SomeModule())
      output = residual_module(input_tensor)
        ```
- **卷积层模块（Conv）**
    - **参数：**
        - `in_channels`(int):输入通道的数量。
        - `out_channels`(int):输出通道的数量。
        - `kernel_size`(input/tuple):卷积核的大小。如果是 tuple，则为 (height, width)。
        - `stride`(input/tuple):步幅大小。默认为1。
        - `bias`(bool，可选):是否包含偏置项，默认为True。
        - `device`(str,可选):张量所在的设备，例如"cuda"或"cpu"；默认为None。
        - `dtype`(str,可选):张量的数据类型，例如"float32"。默认为"float32"。
    - **计算：**
        - 进行二维卷积操作，支持padding=same。
    - **使用方法：**
        ```python
        from mytorch.nn import Conv

      # 示例创建Conv模块
      conv_module = Conv(in_channels=3, out_channels=64, kernel_size=3)
      output = conv_module(input_tensor)
        ```
- **带批归一化的卷积层模块（ConvBN）**
    - **参数：**
        - 继承自`Conv`模块，无额外参数
    - **计算：**
        - 在卷积操作后应用批归一化和ReLU激活函数。
    - **使用方法：**
        ```python
       from mytorch.nn import ConvBN

      # 示例创建ConvBN模块
      conv_bn_module = ConvBN(in_channels=3, out_channels=64, kernel_size=3)
      output = conv_bn_module(input_tensor)
        ```