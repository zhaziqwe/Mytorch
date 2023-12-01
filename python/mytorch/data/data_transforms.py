import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        水平翻转图像，图像表示为大小为 n H x W x C 的 NDArray。
        参数：
            img: H x W x C NDArray 表示的图像
        返回：
            H x W x C 的 ndarray,根据概率 self.p 进行翻转后的图像
        """

        flip_img = np.random.rand() < self.p
        if flip_img:
            img = img[:, ::-1, :]
        return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """
        水平翻转图像，图像表示为大小为 n H x W x C 的 NDArray。
        参数:
            img: 图像的 H x W x C NDArray
        返回:
            对应于以概率 self.p 进行翻转的 H x W x C ndarray 图像
        """

        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        H, W, C = img.shape
        img_pad = np.zeros((H+2*self.padding, W+2*self.padding, C))
        img_pad[self.padding:H+self.padding, self.padding:W+self.padding,:] = img
        img_crop = img_pad[self.padding+shift_x:H+self.padding+shift_x, self.padding+shift_y:W+self.padding+shift_y, :]
        return img_crop