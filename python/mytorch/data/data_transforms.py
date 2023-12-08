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
            img: N x C x H x W NDArray 表示的图像
        返回：
            N x C x H x W 的 ndarray,根据概率 self.p 进行翻转后的图像
        """

        flip_img = np.random.rand() < self.p
        if flip_img:
            img = img[:,:,:, ::-1]
        return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """
        随机裁剪图像，图像表示为大小为 n H x W x C 的 NDArray。
        参数:
            img: 图像的 N x C x H x W  NDArray
        返回:
            N x C x H x W 的 ndarray,根据padding裁剪后的图像
        """

        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        N, C ,H, W, = img.shape
        img_pad = np.zeros((N,C,H+2*self.padding, W+2*self.padding))
        img_pad[:,:,self.padding:H+self.padding, self.padding:W+self.padding] = img
        img_crop = img_pad[:,:,self.padding+shift_x:H+self.padding+shift_x, self.padding+shift_y:W+self.padding+shift_y]
        return img_crop
    
class Normalize(Transform):
    def __init__(self, mean, std):
        self.mean = np.array(mean).reshape(1, 3, 1, 1)
        self.std = np.array(std).reshape(1, 3, 1, 1)

    def __call__(self, img):
        """
        对图像进行归一化处理，使用给定的均值和标准差。
        参数:
            img: 图像的 H x W x C NDArray
        返回:
            归一化后的图像，使用指定的均值和标准差
        """
        img = img.astype(np.float32) / 255.0  # 将图像转换为浮点数，并归一化到 [0, 1] 范围
        img -= self.mean  
        img /= self.std  
        return img

