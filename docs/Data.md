# 数据读取

## Dataset 类
`Dataset` 类是一个代表数据集的抽象类。所有的子类应该按需重写 `__getitem__` 方法，以支持根据给定键获取数据样本。同时子类也必须重写 `__len__` 方法，该方法应返回数据集的大小。

#### 方法：

- **`__init__(self, transforms: Optional[List] = None)`**
    - 作用：初始化 `Dataset` 类的对象。
    - 参数：
        - `transforms`: 可选参数，表示数据集的转换方法列表。
    - 示例：
        ```python
        dataset = Dataset(transforms=[transform1, transform2])
        ```

- **`__getitem__(self, index) -> object`**
    - 作用：根据索引获取数据集中特定位置的数据样本。
    - 参数：
        - `index`: 索引值，用于定位数据样本。
    - 注意：这个方法应该在子类中被实现。

- **`__len__(self) -> int`**
    - 作用：返回数据集的大小。
    - 返回值：数据集中的样本数量。
    - 注意：这个方法应该在子类中被实现。

- **`apply_transforms(self, x)`**
    - 作用：对数据应用预先定义的转换方法。
    - 参数：
        - `x`: 待转换的数据。
    - 返回值：转换后的数据。
    - 示例：
        ```python
        transformed_data = dataset.apply_transforms(data)
        ```

### MNISTDataset 类

`MNISTDataset` 类是用于处理MNIST数据集的类。它继承自抽象类 `Dataset`，并提供了用于加载MNIST数据集的方法和功能。

#### 方法：

- **`__init__(self, image_filename: str, label_filename: str, transforms: Optional[List] = None)`**
    - 作用：初始化 `MNISTDataset` 类的对象，加载图像和标签数据。
    - 参数：
        - `image_filename`: 图像文件的路径。
        - `label_filename`: 标签文件的路径。
        - `transforms`: 可选参数，表示数据集的转换方法列表。
    - 示例：
        ```python
        mnist_dataset = MNISTDataset('train-images.gz', 'train-labels.gz', transforms=[transform1, transform2])
        ```

- **`__getitem__(self, index) -> object`**
    - 作用：根据索引获取数据集中特定位置的数据样本。
    - 参数：
        - `index`: 索引值，用于定位数据样本。
    - 返回值：包含图像数据和标签的元组。
    - 示例：
        ```python
        img, label = mnist_dataset[0]  # 获取第一个数据样本
        ```
        
- **`__len__(self) -> int`**
    - 作用：返回数据集的大小。
    - 返回值：数据集中的样本数量。
    - 示例：
        ```python
        dataset_length = len(mnist_dataset)  # 获取数据集的大小
        ```
