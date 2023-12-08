import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Dataset:
    r"""代表 `Dataset` 的抽象类。
    所有的子类应该重写 :meth:`__getitem__` 方法，支持根据给定的键获取数据样本。
    子类也必须重写 :meth:`__len__` 方法，该方法应返回数据集的大小。
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x

class DataLoader:
    r"""
    数据加载器。结合了数据集和采样器，并提供了对给定数据集的可迭代方式加载数据。
    
    参数:
    dataset (Dataset): 用于加载数据的数据集。
    batch_size (int, 可选): 每批加载多少样本（默认为 ``1``）。
    shuffle (bool, 可选): 设置为 ``True`` 表示每个周期重新打乱数据（默认为 ``False``）。
    """

    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                        range(batch_size, len(dataset), batch_size))
            
        

    def __iter__(self):
        self.idx = 0
        if self.shuffle:
            tmp_range = np.arange(len(self.dataset))
            np.random.shuffle(tmp_range)
            self.ordering = np.array_split(tmp_range,
                range(self.batch_size, len(self.dataset), self.batch_size))
        return self

    def __next__(self):
        if self.idx < len(self.ordering):
            data = self.dataset[self.ordering[self.idx]]
            self.idx += 1
            return [Tensor(x) for x in data]
        else:
            raise StopIteration

    def __len__(self):
        return len(self.dataset)