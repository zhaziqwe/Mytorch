import cupy
import numpy
from .array_device import cpu
from .array_device import gpu

class array_api:
    
    @staticmethod
    def _auto_select_backend(a, b=None):
        if type(a).__module__ == cupy.__name__:
            return cupy
        elif type(a).__module__ == numpy.__name__:
            return numpy
        else:
            raise ValueError("Unknown or mismatched array types for backend")
        
    @staticmethod
    def add(a, b):
        return array_api._auto_select_backend(a, b).add(a, b)
    
    @staticmethod
    def where(condition, x, y):
        return array_api._auto_select_backend(condition).where(condition, x, y)

    @staticmethod
    def multiply(a, b):
        return array_api._auto_select_backend(a, b).multiply(a, b)
    
    @staticmethod
    def divide(a, b):
        return array_api._auto_select_backend(a, b).divide(a, b)
    
    @staticmethod
    def power(a, exponent):
        return array_api._auto_select_backend(a).power(a, exponent)
    
    @staticmethod
    def transpose(a, axes=None):
        return array_api._auto_select_backend(a).transpose(a, axes)
    
    @staticmethod
    def reshape(a, newshape):
        return array_api._auto_select_backend(a).reshape(a, newshape)
    
    @staticmethod
    def sum(a, axis=None, keepdims=False):
        return array_api._auto_select_backend(a).sum(a, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def broadcast_to(a, shape):
        return array_api._auto_select_backend(a).broadcast_to(a, shape)
    
    @staticmethod
    def matmul(a, b):
        return array_api._auto_select_backend(a, b).matmul(a, b)
    
    @staticmethod
    def log(a):
        return array_api._auto_select_backend(a).log(a)
    
    @staticmethod
    def exp(a):
        return array_api._auto_select_backend(a).exp(a)
    
    @staticmethod
    def negative(a):
        return array_api._auto_select_backend(a).negative(a)
    
    @staticmethod
    def maximum(a, b):
        return array_api._auto_select_backend(a, b).maximum(a, b)
    
    @staticmethod
    def max(a, axis=None, keepdims=False):
        return array_api._auto_select_backend(a).max(a, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def minimum(a, b):
        return array_api._auto_select_backend(a, b).minimum(a, b)
    
    @staticmethod
    def min(a, axis=None, keepdims=False):
        return array_api._auto_select_backend(a).min(a, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def abs(a):
        return array_api._auto_select_backend(a).abs(a)
    
    @staticmethod
    def sqrt(a):
        return array_api._auto_select_backend(a).sqrt(a)
    
    @staticmethod
    def random_normal(shape):
        return array_api._auto_select_backend(shape).random.normal(shape)
    
    @staticmethod
    def random_uniform(shape):
        return array_api._auto_select_backend(shape).random.uniform(shape)
    
    @staticmethod
    def as_strided(a, shape, strides):
        return array_api._auto_select_backend(a).lib.stride_tricks.as_strided(a, shape, strides)
    
    @staticmethod
    def pad(a, pad_width, mode='constant', **kwargs):
        return array_api._auto_select_backend(a).pad(a, pad_width, mode=mode, **kwargs)
    
    @staticmethod
    def flip(a, axis):
        return array_api._auto_select_backend(a).flip(a, axis)
    
    @staticmethod
    def tanh(a):
        return array_api._auto_select_backend(a).tanh(a)

    """
        以上全部是计算函数，下面写生成函数
    """

    @staticmethod
    def ones(shape, dtype = "float32", device = cpu()):
        if device == cpu():
            return numpy.ones(shape, dtype = dtype)
        elif device == gpu():
            return cupy.ones(shape, dtype = dtype)
        else:
            raise ValueError("Unknown or mismatched array types for backend")
    
    @staticmethod
    def zeros(shape, dtype = "float32", device = cpu()):
        if device == cpu():
            return numpy.zeros(shape, dtype = dtype)
        elif device == gpu():
            return cupy.zeros(shape, dtype = dtype)
        else:
            raise ValueError("Unknown or mismatched array types for backend")
        
    @staticmethod
    def rand(*shape, device = cpu()):
        if device == cpu():
            return numpy.random.rand(*shape)
        elif device == gpu():
            return cupy.random.rand(*shape)
        else:
            raise ValueError("Unknown or mismatched array types for backend")
        
    @staticmethod
    def randn(*shape, device = cpu()):
        if device == cpu():
            return numpy.random.randn(*shape)
        elif device == gpu():
            return cupy.random.randn(*shape)
        else:
            raise ValueError("Unknown or mismatched array types for backend")
        
    @staticmethod
    def one_hot(n, i, dtype = "float32", device = cpu()):
        if device == cpu():
            return numpy.eye(n, dtype = dtype)[i]
        elif device == gpu():
            return cupy.eye(n, dtype = dtype)[i]
        else:
            raise ValueError("Unknown or mismatched array types for backend")
    

NDArray = None

