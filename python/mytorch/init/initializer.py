import math
import mytorch as torch 
from mytorch.array_api import array_api

"""
    一些初始化向量的函数
"""

def rand(*shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False):
    device = torch.cpu() if device is None else device
    array = array_api.rand(*shape, device=device) * (high - low) + low
    return torch.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def randn(*shape, mean=0.0, std=1.0, device=None, dtype="float32", requires_grad=False):
    device = torch.cpu() if device is None else device
    array = array_api.randn(*shape, device=device) * std + mean
    return torch.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    device = torch.cpu() if device is None else device
    array = array_api.ones(shape, device=device, dtype=dtype) * c
    return torch.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def ones(*shape, device=None, dtype="float32", requires_grad=False):
    return constant(
        *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def zeros(*shape, device=None, dtype="float32", requires_grad=False):
    return constant(
        *shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False):
    """Generate binary random Tensor"""
    device = torch.cpu() if device is None else device
    array = array_api.rand(*shape, device=device) <= p
    return torch.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def one_hot(n, i, device=None, dtype="float32", requires_grad=False):
    """Generate one-hot encoding Tensor"""
    device = torch.cpu() if device is None else device
    return torch.Tensor(
        array_api.one_hot(n, i.realize_cached_data(), dtype=dtype, device=device),
        device=device,
        requires_grad=requires_grad,
    )


def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", device=None, **kwargs):
    # print(device)
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3 / fan_in)
    if shape is not None:
        return rand(*shape, low=-bound, high=bound, device=device, **kwargs)
    else:
        return rand(fan_in, fan_out, low=-bound, high=bound, device=device,**kwargs)

