# 神经网络模块

mytorch里提供了丰富的网路模块供炼丹师们训练，下面我们介绍一下这些模块的作用。

- **全连接层（Fully Connected Layer）**
    - **参数：**
        - `in_features`: 输入特征的数量。
        - `out_features`: 输出特征的数量。
    - **计算：**
        - 进行输入和权重的矩阵乘法，然后加上偏置项。
    - **使用方法：**
        ```python
        import mytorch.nn as nn
        
        # 示例创建全连接层
        fc_layer = nn.Linear(in_features=100, out_features=50)
        ```