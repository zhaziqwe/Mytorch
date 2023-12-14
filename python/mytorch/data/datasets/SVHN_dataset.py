
import numpy as np
from scipy.io import loadmat
from typing import List, Optional, Iterable
from ..data_basic import Dataset

class SVHNDataset(Dataset):
    def __init__(
        self,
        file: str,
        transforms: Optional[List] = None,
    ):
        super(SVHNDataset, self).__init__(transforms)
        # 加载训练数据
        data = loadmat(file)
        self.images = data['X']
        self.labels = data['y'].flatten()

    def __getitem__(self, index) -> object:
        # 根据给定的索引获取数据样本
        if isinstance(index, (Iterable, slice)):
            img = np.stack([i.reshape((1, 32, 32)) for i in self.images[index]])
        else:
            img = self.images[index].reshape((1, 1, 32, 32))
        img = self.apply_transforms(img)
        label = self.labels[index]
        return img, label

    def __len__(self) -> int:
        # 返回数据集的大小
        return len(self.images)



