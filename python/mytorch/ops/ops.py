from typing import Optional, Tuple, Union
from ..autograd import Tensor, Value, TensorOp
from ..array_api import array_api, NDArray
import numpy
import cupy
from ..array_device import cpu
from ..array_device import gpu
"""
    所有的计算函数的compute参数传入的都是NDArry类型
    他是通过make_from_op调用的realize_cached_data方法调用的
    realize_cached_data中会把tensor中的array提取出来
"""

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Tensor的整数幂"""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return  out_grad * self.scalar * power_scalar(a, self.scalar-1)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """逐元素的除法"""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad * (power_scalar(rhs,-1)) , out_grad * (divide(negate(lhs),power_scalar(rhs,2)))


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return (out_grad / self.scalar,)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        self.axis_l = list(range(a.ndim))
        if self.axes is None:
            self.axis_l[-1], self.axis_l[-2] = self.axis_l[-2], self.axis_l[-1]
        else:
            self.axis_l[self.axes[0]], self.axis_l[self.axes[1]] = self.axes[1], self.axes[0]
            
        return array_api.transpose(a, self.axis_l)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return reshape(out_grad, a.shape)

def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # print(a.shape)
        # print(self.shape)
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        out_grad_shape = out_grad.shape
        self.reduce_dim = []
        # 接下来为了找到reduce_dim我们需要有指针比较,指针指向的是dim,指向负数表示超出shape范围
        point_in = len(input_shape) - 1
        # 广播后的shape肯定更大，所以我们遍历out_grad_shape比较方便
        for point_out in range(len(out_grad_shape) - 1,-1,-1):
            # 如果超出范围则是需要reduce的dim
            if point_in < 0:
                self.reduce_dim.append(point_out)
                continue
            # 比较广播后对应dim的大小是否相等，不等也是需要reduce的dim
            if  self.shape[point_out]!= input_shape[point_in]:
                self.reduce_dim.append(point_out)
            # 左移in的指针
            point_in -= 1
        # 转换传入参数类型
        out_grad = summation(out_grad,tuple(self.reduce_dim))
        return reshape(out_grad,input_shape)



def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        if isinstance(self.axes, int): 
            self.axes = tuple([self.axes])
    
    def compute(self, a):

        return array_api.sum(a, self.axes)


    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        # 计算出input执行sum的轴再广播后（变成1）的shape
        final_shape = list(input_shape)
        
        if self.axes:
            for dim in self.axes:
                final_shape[dim] = 1
        else:
            final_shape = [1 for _ in range(len(final_shape))]
        out_grad = reshape(out_grad,final_shape)
        return out_grad * array_api.ones(input_shape, dtype = "float32", device = out_grad.device)



def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        dlhs = matmul(out_grad,transpose(rhs))
        drhs = matmul(transpose(lhs),out_grad)
        if dlhs.shape != lhs.shape:
            dlhs = summation(dlhs, tuple(range(len(dlhs.shape) - len(lhs.shape))))
        if drhs.shape != rhs.shape:
            drhs = summation(drhs, tuple(range(len(drhs.shape) - len(rhs.shape))))
        return dlhs , drhs


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return negate(out_grad)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad * power_scalar(a,-1)


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad * exp(a)


def exp(a):
    return Exp()(a)

class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        max_z = array_api.max(Z, axis=self.axes, keepdims=False)
        logsumexp = array_api.log(array_api.sum(array_api.exp(Z - max_Z),axis=self.axes)) + max_z
        return logsumexp 

    def gradient(self, out_grad, node):
        Z = node.inputs[0]
        max_Z = array_api.max(Z.cached_data, axis=self.axes, keepdims=True)
        exp_val = exp(Z - Tensor(max_Z))
        sum_val = summation(exp_val, axes=self.axes)

        log_grad = out_grad / sum_val
        
        input_shape = node.inputs[0].shape
        final_shape = list(input_shape)
        if self.axes:
            if isinstance(self.axes, int):
                final_shape[self.axes] = 1
            else:
                for dim in self.axes:
                    final_shape[dim] = 1
        else:
            final_shape = [1 for _ in range(len(final_shape))]
        sum_grad = reshape(log_grad, tuple(final_shape))
        sum_grad_b = broadcast_to(sum_grad, Z.shape)
        return exp_val * sum_grad_b

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Softmax(TensorOp):
    def __init__(self, axis: int):
        self.axis = axis

    def compute(self, a):
        exp_a = array_api.exp(a - array_api.max(a, axis=self.axis, keepdims=True))
        return exp_a / array_api.sum(exp_a, axis=self.axis, keepdims=True)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        softmax_out = softmax(a, axis=self.axis)
        return out_grad * (softmax_out * (1 - softmax_out))


def softmax(a, axis):
    return Softmax(axis)(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        relu_mask = array_api.where(node.inputs[0].cached_data > 0, 1.0, 0.0).astype("float32")
        return out_grad * relu_mask


def relu(a):
    return ReLU()(a)


class GeLU(TensorOp):
    def compute(self, a):
        # GELU activation function: x * 0.5 * (1.0 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
        sqrt2_over_pi = array_api.sqrt(2 / array_api.pi)
        x_cubed = a ** 3
        tanh_term = array_api.tanh(sqrt2_over_pi * (a + 0.044715 * x_cubed))
        return a * 0.5 * (1.0 + tanh_term)

    def gradient(self, out_grad, node):
        a = node.inputs[0].cached_data
        # Compute the gradient of GELU w.r.t. its input
        sqrt2_over_pi = array_api.sqrt(2 / array_api.pi)
        x_cubed = a ** 3
        tanh_term = array_api.tanh(sqrt2_over_pi * (a + 0.044715 * x_cubed))
        grad_gelu = 0.5 * (1.0 + tanh_term) + a * (1.0 - tanh_term ** 2) * sqrt2_over_pi * (1 + 0.044715 * x_cubed)
        return out_grad * grad_gelu

def gelu(a):
    return GeLU()(a)



class Sigmoid(TensorOp):
    def compute(self, a):
        return 1 / (1  + array_api.exp(-a))

    def gradient(self, out_grad, node):
        return out_grad * sigmoid(node.inputs[0]) * (1 - sigmoid(node.inputs[0]))


def sigmoid(a):
    return Sigmoid()(a)


class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        input = node.inputs[0]
        out_grad = out_grad * (
            1 + (-tanh(input) ** 2)
        )
        return out_grad


def tanh(a):
    return Tanh()(a)


class Sqrt(TensorOp):
    def compute(self, a):
        return array_api.sqrt(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0].cached_data
        # Gradient of sqrt(x) is 1 / (2 * sqrt(x))
        sqrt_grad = 1.0 / (2.0 * array_api.sqrt(a))
        return out_grad * sqrt_grad

def sqrt(a):
    return Sqrt()(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.flip(a, self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes)


def flip(a, axes):
    return Flip(axes)(a)


def compact(array):
    out_array = array.copy()
    return out_array

class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        shape = a.shape
        out_shape = list(shape)
        slices = [slice(0, out_shape[idx]) for idx in range(len(shape))]
        for ax in self.axes:
            if ax >= len(out_shape):
                continue
            out_shape[ax] = out_shape[ax] * (1 + self.dilation)
            slices[ax] = slice(0, out_shape[ax], 1 + self.dilation)
        if isinstance(a, (numpy.generic,numpy.ndarray)):
            device = cpu()
        elif isinstance(a, (cupy.generic,cupy.ndarray)):
            device =  gpu()
        out_tensor = array_api.zeros(out_shape, dtype = "float32", device = device) 
        # 使用切片索引将输入数组 a 复制到 out_tensor 中指定的位置
        out_tensor[tuple(slices)] = a
        return out_tensor

    def gradient(self, out_grad, node):
        out_grad = undilate(out_grad, self.axes, self.dilation)
        return out_grad

def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)



class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        shape = a.shape
        slices = [slice(0, shape[idx]) for idx in range(len(shape))]
        for ax in self.axes:
            if ax >= len(shape):
                continue
            slices[ax] = slice(0, shape[ax], 1 + self.dilation)
        return compact(a[tuple(slices)])

    def gradient(self, out_grad, node):
        out_grad = dilate(out_grad, self.axes, self.dilation)
        return out_grad


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)



class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        N, H, W, C_in = A.shape
        K, _, I, C_out = B.shape
        assert C_in == I, "input tensor shape and kernel dosen't match"
        
        pad_width = [(0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                    (0, 0)]
        _A = array_api.pad(A, pad_width, mode='constant', constant_values=0) if self.padding > 0 else A

        inner_dim = K * K * C_in
        Ns, Hs, Ws, Cs = _A.strides
        H_out = (H - K + 2 * self.padding) // self.stride + 1
        W_out = (W - K + 2 * self.padding) // self.stride + 1

        _A = array_api.as_strided(
            _A,
            shape=(N, H_out, W_out, K, K, C_in),
            strides=(Ns, Hs*self.stride, Ws*self.stride, Hs, Ws, Cs)
        )
        _A_ = compact(_A).reshape((-1, inner_dim))
        _B_ = compact(B).reshape((-1, C_out))
        # print(_A_.sum())
        # print(_B_.sum())
        out = _A_ @ _B_
        # print(out.sum())
        return out.reshape((N, H_out, W_out, C_out))
    
    def gradient(self, out_grad: Value, node: Value):
        input, weight = node.inputs

        #对input的导数
        #根据原理图，需要先padding一圈，但是我们的conv自带了这个操作，所以先考虑dilate操作
        # N H W C 所以是 1, 2 轴
        grad_dilate = dilate(out_grad,(1, 2), self.stride - 1)
        weight_r180 = flip(weight, (0, 1))
        weight_t = transpose(weight_r180)
        K = weight_r180.shape[0]
        grad_input = conv(grad_dilate, weight_t, 1, K - 1 - self.padding)

        #对W的导数 需要更换索引顺序
        grad_dilate = grad_dilate.transpose((0, 2)).transpose((0, 1))
        input_t = transpose(input,(0,3))
        grad_weight = conv(input_t, grad_dilate, 1, self.padding)
        grad_weight = grad_weight.transpose((0, 2)).transpose((0, 1))

        return Tensor(grad_input), grad_weight



def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


class Max(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        if isinstance(self.axes, int): 
            self.axes = tuple([self.axes])

    def compute(self, a):
        # 计算最大值，同时保持维度以便广播操作
        return array_api.max(a, axis=self.axes, keepdims=True)

    def gradient(self, out_grad, node):
        a = node.inputs[0]  # 假设node是一个封装了数据的对象
        array = a.cached_data
        # 计算最大值，保持维度以便广播操作
        max_values = self.compute(array)
        # 生成一个布尔数组，其中最大值位置为True
        mask = (array== max_values)
        # 计算输入梯度的总和
        grad_sum = array_api.sum(mask, axis=self.axes, keepdims=True)
        # 将外部梯度（out_grad）广播到与输入数据相同的形状，然后乘以mask。
        # 这样，只有最大值位置才会获得梯度值。最后，我们需要将梯度值平均分配到所有最大值位置，
        # 这是通过除以grad_sum实现的，以处理存在多个最大值的情况。
        grad = out_grad * mask / grad_sum
        return grad

def max(a, axes=None):
    return Max(axes)(a)


class MaxPooling2D(TensorOp):
    def __init__(self,kernel_size: int, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self._A_precomputed = None
    
    def precompute(self, A):
        N, H, W, C = A.shape
        K = self.kernel_size
        stride = self.stride
        padding = self.padding
        
        # 应用填充
        pad_width = [(0, 0), (0, 0), (padding, padding), (padding, padding)]
        _A = array_api.pad(A, pad_width, mode='constant', constant_values=0) if self.padding > 0 else A
        
        # 计算输出形状
        H_out = (H - K + 2 * padding) // stride + 1
        W_out = (W - K + 2 * padding) // stride + 1
        
        # 获取每个窗口的视图
        Ns, Hs, Ws, Cs = _A.strides
        _A = array_api.as_strided(
            _A,
            shape=(N, H_out, W_out, K, K, C),
            strides=(Ns, Hs*stride, Ws*stride, Hs, Ws, Cs)
        )
        
        self._A_precomputed = compact(_A)  # 存储预计算的值
        
    
    def compute(self, A):
        self.precompute(A)
        # 计算每个窗口的最大值
        max_pool_out = array_api.max(self._A_precomputed, axis=(3, 4),keepdims=False)
        return max_pool_out
        
    def gradient(self, out_grad: Tensor, node: Tensor):
        
        array = self._A_precomputed
        array = array.transpose((0,1,2,5,4,3))
        N,H,W,C,K,_ = array.shape
        
        argmax = array_api.argmax(array.reshape(N,H,W,C,K**2), axes=-1)
        grad_input = array_api.zeros_like(node.inputs[0].cached_data)
        
        if isinstance(grad_input, (numpy.generic,numpy.ndarray)):
            for index in numpy.ndindex(argmax.shape):
                n, h, w, c = index
                max_index = argmax[index]
                mh, mw = max_index // K, max_index % K
                grad_input[n, h * self.stride + mh, w * self.stride + mw, c] += out_grad.cached_data[index]
        else:
            for index in cupy.ndindex(argmax.shape):
                n, h, w, c = index
                max_index = argmax[index]
                mh, mw = max_index // K, max_index % K
                grad_input[n, h * self.stride + mh, w * self.stride + mw, c] += out_grad.cached_data[index]
    
        return Tensor(grad_input)
       
def maxPooling2D(a,kernel_size,stride=1,padding=0):
    return MaxPooling2D(kernel_size,stride, padding)(a)


class Embedding(TensorOp):
    def __init__(self, x):
        self.x = x.cached_data
    
    def compute(self, weight):
        return weight[self.x.flatten()]

    def gradient(self, out_grad, node):
        weight = node.inputs[0].cached_data
        grad_weight = array_api.zeros_like(weight)
        array_api.add_at(grad_weight, self.x.flatten(), out_grad.cached_data)
        return Tensor(grad_weight)

def embedding(x, weight):
    return Embedding(x)(weight)