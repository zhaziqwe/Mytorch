from typing import List, Any
from mytorch.autograd import Tensor
from mytorch import ops
from mytorch import init
from mytorch import array_api
import math

class Parameter(Tensor):
    """一种特殊的Tensor参数, 需要更新的参数"""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        # 这种情况只会出现在Sequential中
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


"""
    目前这个函数主要服务于module的train和eval状态的切换
"""

def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def print(self):
        print(self.__dict__)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(init.kaiming_uniform(
            in_features, out_features, device=device, dtype=dtype))
        self.bias = Parameter(ops.transpose(init.kaiming_uniform(
            out_features, 1, device=device, dtype=dtype))) if bias else None


    def forward(self, X: Tensor) -> Tensor:
        Y = ops.matmul(X, self.weight)
        if self.bias:
            bias = ops.broadcast_to(self.bias, Y.shape)
            Y += bias
        return Y


class Flatten(Module):
    def forward(self, X):
        return ops.reshape(X, (X.shape[0], -1))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        batch_size = logits.shape[0]
        classes = logits.shape[1]
        normalize_x = ops.logsumexp(logits, axes=1)
        y_one_hot = init.one_hot(classes, y, device=y.device)
        Z_y = ops.summation(logits * y_one_hot,axes=1) 
        
        loss = ops.summation(normalize_x - Z_y)
        return loss / batch_size


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype)
        self.running_var = init.ones(self.dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        mean = x.sum((0, )) / batch_size
        # NOTE 广播前reshape
        x_minus_mean = x - mean.reshape((1, x.shape[1])).broadcast_to(x.shape)
        var = (x_minus_mean ** 2).sum((0, )) / batch_size
        
        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data

            x_std = ((var + self.eps) ** 0.5).reshape((1, x.shape[1])).broadcast_to(x.shape)
            x_normed = x_minus_mean / x_std
            return x_normed * self.weight.reshape((1, self.dim)).broadcast_to(x.shape) + self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        else:
            # NOTE 没有动量机制了
            x_normed = (x - self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)) / (self.running_var.reshape((1, self.dim)).broadcast_to(x.shape) + self.eps) ** 0.5
            # NOTE 测试时依旧需要W和B
            return x_normed * self.weight.reshape((1, self.dim)).broadcast_to(x.shape) + self.bias.reshape((1, self.dim)).broadcast_to(x.shape)




class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))

class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(Tensor([1.0] * dim, device=device, dtype=dtype))
        self.bias = Parameter(Tensor([0.0] * dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        features = x.shape[-1]

        mean_x = ops.divide_scalar(ops.summation(x, axes=-1), features)
        broadcast_mean = ops.broadcast_to(
            ops.reshape(mean_x, (-1, 1)), x.shape)

        numerator = x - broadcast_mean

        var_x = ops.power_scalar(numerator, 2)
        var_x = ops.summation(ops.divide_scalar(var_x, features), axes=-1)
        broadcast_var = ops.broadcast_to(ops.reshape(var_x, (-1, 1)), x.shape)

        denominator = ops.power_scalar(broadcast_var+self.eps, 0.5)

        frac = numerator / denominator

        broadcast_weight = ops.broadcast_to(
            ops.reshape(self.weight, (1, -1)), x.shape)
        broadcast_bias = ops.broadcast_to(
            ops.reshape(self.bias, (1, -1)), x.shape)
        y = ops.multiply(broadcast_weight, frac) + broadcast_bias
        return y

class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(Tensor([1.0] * dim, device=device, dtype=dtype))
        self.bias = Parameter(Tensor([0.0] * dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        # x 的形状为 (batch_size, sequence_length, embedding_dim)
        batch_size, sequence_length, embedding_dim = x.shape

        # 计算每个样本在 embedding_dim 维度上的均值和方差
        mean_x = ops.divide_scalar(ops.summation(x, axes=-1), embedding_dim)
        mean_x = ops.broadcast_to(ops.reshape(mean_x, (batch_size, sequence_length, 1)), x.shape)

        numerator = x - mean_x

        var_x = ops.power_scalar(numerator, 2)
        var_x = ops.summation(ops.divide_scalar(var_x, embedding_dim), axes=-1)
        var_x = ops.broadcast_to(ops.reshape(var_x, (batch_size, sequence_length, 1)), x.shape)

        denominator = ops.power_scalar(var_x + self.eps, 0.5)

        frac = numerator / denominator

        # 应用权重和偏置
        broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, 1, -1)), x.shape)
        broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1, 1, -1)), x.shape)
        y = ops.multiply(broadcast_weight, frac) + broadcast_bias
        return y




class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(*x.shape, p=1-self.p, dtype='float32', device=x.device)
            x = x * mask
            z = x/(1-self.p)
        else:
            z = x
        return z


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x


class Conv(Module):
    """
    多通道的2D卷积层
    重要提示：接受 NCHW 格式的输入，输出也是 NCHW 格式
    仅支持 padding=same
    不支持分组卷积或扩张
    仅支持方形核心大小
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device
        self.dtype = dtype

        self.padding = self.kernel_size // 2
        self.weight = Parameter(
            init.kaiming_uniform(
                kernel_size * kernel_size * in_channels,
                kernel_size * kernel_size * out_channels,
                shape=[kernel_size, kernel_size, in_channels, out_channels],
                dtype=dtype,
                device=device
            )
        )
        self.bias = None
        if bias:
            prob = 1.0 / (in_channels * kernel_size ** 2) ** 0.5
            self.bias = Parameter(
                init.rand(
                    out_channels,
                    low=-prob,
                    high=prob,
                    device=device,
                    dtype=dtype
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        x = ops.transpose(
            ops.transpose(x),
            (1, 3)
        ) # NCHW -> NCWH -> NHWC
        x = ops.conv(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            x = x + ops.broadcast_to(self.bias, x.shape)
        x = ops.transpose(
            ops.transpose(x, (1,3))
        ) # NHWC -> NCWH -> NCHW
        return x
    

class ConvBN(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride, bias, device, dtype)
        self.bn = BatchNorm2d(out_channels, device=device, dtype=dtype)
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))
    


class MaxPooling2D(Module):
    """
    2D最大池化层
    接受 NCHW 格式的输入，输出也是 NCHW 格式
    """
    def __init__(self, kernel_size, stride=1, padding=0, device=None, dtype="float32"):
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.device = device
        self.dtype = dtype

    def forward(self, x: Tensor) -> Tensor:
        x = ops.transpose(
            ops.transpose(x),
            (1, 3)
        ) # NCHW -> NCWH -> NHWC
        x = ops.maxPooling2D(x,self.kernel_size,self.stride,self.padding)
        x = ops.transpose(
            ops.transpose(x, (1,3))
        ) # NHWC -> NCWH -> NCHW
        return x


class Embedding(Module):
    """
    词嵌入层
    接受整数序列，输出为词向量序列
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, device=None, dtype="float32"):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = Parameter(init.rand(num_embeddings, embedding_dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        if self.padding_idx is not None:
            mask = x == self.padding_idx
            x = x * (1 - mask) + 0 * mask
        return ops.embedding(x, self.weight)


class CausalSelfAttention(Module):

    """
    多头注意力机制模块
    """
    def __init__(self, max_len, n_embd, n_head, device=None, dtype="float32"):
        assert n_embd % n_head == 0, "n_embd should be divisible by n_head"
        self.n_embd = n_embd
        self.n_head = n_head

        self.q_linear = Linear(n_embd, n_embd, bias=False, device=device, dtype=dtype)
        self.k_linear = Linear(n_embd, n_embd, bias=False, device=device, dtype=dtype)
        self.v_linear = Linear(n_embd, n_embd, bias=False, device=device, dtype=dtype)
        self.out_linear = Linear(n_embd, n_embd, bias=False, device=device, dtype=dtype)
        self.mask = array_api.trils(array_api.ones((max_len, max_len), device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, n_embd = x.shape
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        head_size = n_embd // self.n_head
        q = ops.reshape(q, (batch_size, seq_len, self.n_head, head_size))     
        k = ops.reshape(k, (batch_size, seq_len, self.n_head, head_size))
        v = ops.reshape(v, (batch_size, seq_len, self.n_head, head_size))
        # B, HN, S, HS
        q = ops.transpose(q, (2, 1))
        k = ops.transpose(k, (2, 1))
        v = ops.transpose(v, (2, 1))
        # B, HN, HS, S
        k = ops.transpose(k, (2, 3))
        # B, HN, S, S
        attn_weights = ops.matmul(q, k) * (1.0 / math.sqrt(head_size))
        mask_x = Tensor(self.mask[:seq_len, :seq_len])
        attn_weights = attn_weights * ops.broadcast_to(mask_x, attn_weights.shape) + (1.0 - mask_x) * -1e9
        attn_weights = ops.softmax(attn_weights, axis=-1)
        # B, HN, S, HS
        attn = ops.matmul(attn_weights, v)
        # B, S, HN, HS
        attn = ops.transpose(attn, (2, 1))
        # B, S, HN*HS
        attn = ops.reshape(attn, (batch_size, seq_len, self.n_head * head_size))
        # B, S, E
        output = self.out_linear(attn)
        return output
   


class TransformerDecoderLayer(Module):
    """
    多头自注意力解码层
    """
    def __init__(self, max_len, n_embd, n_head, dropout=0.1, activation="relu", device=None, dtype="float32"):
        if activation == "relu":
            self.activation = ops.relu
        elif activation == "gelu":
            self.activation = ops.gelu
        else:
            raise ValueError("activation should be relu or gelu, not {}".format(activation))
        

        self.self_attn = CausalSelfAttention(max_len, n_embd, n_head, device=device, dtype=dtype)
        self.layer_norm1 = LayerNorm(n_embd, device=device, dtype=dtype)
        self.layer_norm2 = LayerNorm(n_embd, device=device, dtype=dtype)
        self.mlp = Sequential(
            Linear(n_embd, 4 * n_embd, bias=False, device=device, dtype=dtype),
            self.activation,
            Dropout(dropout),
            Linear(4 * n_embd, n_embd, bias=False, device=device, dtype=dtype),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x
    


class TransformerDecoder(Module):
    """
    多头自注意力解码器
    """
    def __init__(self, vocab_size, max_len, n_embd, n_head, n_layer, dropout=0.1, activation="relu", device=None, dtype="float32"):
        super().__init__()
        self.n_embd = n_embd
        self.wte = Embedding(vocab_size, n_embd, device=device, dtype=dtype)
        self.wpe = Embedding(max_len, n_embd, device=device, dtype=dtype)
        self.layers = Sequential(*[TransformerDecoderLayer(max_len, n_embd, n_head, dropout, activation, device, dtype) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd, device=device, dtype=dtype)
        self.lm_head = Linear(n_embd, vocab_size, bias=False, device=device, dtype=dtype)
        # weight sharing
        # self.wte.weight.cached_data = self.lm_head.weight.cached_data.T

    def forward(self, idx: Tensor) -> Tensor:
        batch_size, seq_len = idx.shape
        tok_emb = self.wte(idx)
        tok_emb = ops.reshape(tok_emb, (batch_size, seq_len, self.n_embd))
        pos_emb = self.wpe(Tensor(array_api.arange(seq_len, device=idx.device, dtype=idx.dtype)))
        pos_emb = ops.broadcast_to(pos_emb, (batch_size, seq_len, self.n_embd))

        x = tok_emb + pos_emb
        x = self.layers(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


        

        
