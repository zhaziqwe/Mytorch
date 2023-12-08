
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
        self.images = np.transpose(data['X'], (3, 0, 1, 2))
        self.labels = data['y'].flatten()
        # 根据问题中的说明，对标签进行调整
        self.adjust_labels()

    def adjust_labels(self):
        # 调整标签，使得类别1对应标签1，类别9对应标签9，类别0对应标签10
        self.labels[self.labels == 10] = 0

    def __getitem__(self, index) -> object:
        # 根据给定的索引获取数据样本
        if isinstance(index, (Iterable, slice)):
            img = np.stack([i.reshape((3, 32, 32)) for i in self.images[index]])
        else:
            img = self.images[index].reshape((1, 3, 32, 32))
        img = self.apply_transforms(img)
        label = self.labels[index]
        return img, label

    def __len__(self) -> int:
        # 返回数据集的大小
        return len(self.images)



