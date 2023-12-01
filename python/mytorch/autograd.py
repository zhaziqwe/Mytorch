import numpy
import cupy
from typing import List, Optional, NamedTuple, Tuple, Union, Dict
import mytorch
from mytorch.array_device import *
from mytorch import init
"自动梯度求导"

LAZY_MODE = False


from .array_api import array_api, NDArray


class Op:
    """Operator定义"""

    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args: Tuple[NDArray]):
        """
            前向传播
        """
        raise NotImplementedError()

    def gradient(
        self, out_grad: "Value", node: "Value"
    ) -> Union["Value", Tuple["Value"]]:
        """
            计算梯度
        """
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
        """ 从梯度调用返回元组的便捷方法"""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)


class TensorOp(Op):
    """ 这类op返回的都是tensor """

    def __call__(self, *args):
        return Tensor.make_from_op(self, args)


class Value:
    """计算图中的一个值"""

    # 跟踪计算图
    # 算子操作
    op: Optional[Op]
    # 输入节点
    inputs: List["Value"]
    # 动态计算
    # 存储数据的核心部分(缓存)
    cached_data: NDArray
    requires_grad: bool

    def realize_cached_data(self):
        # 避免重复计算
        if self.cached_data is not None:
            return self.cached_data
        # 注意：数据隐式调用已实现的缓存数据
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        return self.cached_data

    def is_leaf(self):
        return self.op is None

    def _init(
        self,
        op: Optional[Op],
        inputs: List["Tensor"],
        *,
        cached_data: List[object] = None,
        requires_grad: Optional[bool] = None
    ):
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad


class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self,
        array,
        *, 
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            """
                如果我们从一个Tensor复制到这个Tensor里
                我们需要判断指定的这个Tensor类型是否和来源相同
                如果不同，那么需要从来源Tensor中获取数据转换为numpy，然后再转换为指定的Tensor
            """
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                if device == cpu():
                    cached_data = Tensor._array_from_numpy(
                        array.numpy(), dtype=dtype)
                else:
                    cached_data = Tensor._array_from_cupy(
                        array.cupy(), dtype=dtype)
        else: 
            # print(type(array))

            # 直接创建的话保存在cpu中，因为传参是list
            if device is None:
                if isinstance(array, (cupy.generic,cupy.ndarray)):
                    device = gpu()
                else:
                    device = cpu()

            if  device == gpu():
                cached_data = Tensor._array_from_cupy(
                    array, dtype=dtype)
        
            if device == cpu():
                cached_data = Tensor._array_from_numpy(
                    array, dtype=dtype)
        
        

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array, dtype):
        # print ("numpy")
        return numpy.array(numpy_array, dtype=dtype)
    
    @staticmethod
    def _array_from_cupy(cupy_array, dtype):
        # print ("cupy")
        return cupy.array(cupy_array, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        """创建一个共享数据但与图分离的新张量"""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        data = self.realize_cached_data()
        if isinstance(data, (numpy.generic,numpy.ndarray)):
            return cpu()
        elif isinstance(data, (cupy.generic,cupy.ndarray)):
            return gpu()
        else:
            print(type(data))
            raise ValueError("Unsupported data type.")
    
    @device.setter
    def device(self, new_device):
        if new_device not in [cpu(), gpu()]:
            raise ValueError("Unsupported device type.")
        else:
            if new_device == cpu():
                self.cached_data = numpy.array(self.cached_data)
            elif new_device == gpu():
                self.cached_data = cupy.array(self.cached_data)

    def to(self, device):
        if device == 'cpu':
            # 如果要移动到 CPU，将数据转换为 NumPy 数组
            self.device = cpu()
        elif device == 'gpu':
            # 如果要移动到 GPU，将数据转换为 CuPy 数组
            self.device = gpu()
        else:
            raise ValueError("Unsupported device type. Use 'cpu' or 'gpu'.")


    def backward(self, out_grad=None):
        out_grad = out_grad if out_grad else init.ones(
            *self.shape, dtype=self.dtype, device=self.device)
        compute_gradient_of_variables(self, out_grad)

    
    # 重载运算符

    def __repr__(self):
        return "mytorch.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()
        if isinstance(data, (numpy.generic,numpy.ndarray)):
            return data
        return data.numpy()
    
    def cupy(self):
        data = self.realize_cached_data()
        if isinstance(data, (cupy.generic,cupy.ndarray)):
            return data 
        return data.cupy()

    def __add__(self, other):
        if isinstance(other, Tensor):
            return mytorch.ops.EWiseAdd()(self, other)
        else:
            return mytorch.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return mytorch.ops.EWiseMul()(self, other)
        else:
            return mytorch.ops.MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            raise NotImplementedError()
        else:
            return mytorch.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return mytorch.ops.EWiseAdd()(self, mytorch.ops.Negate()(other))
        else:
            return mytorch.ops.AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return mytorch.ops.EWiseDiv()(self, other)
        else:
            return mytorch.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return mytorch.ops.MatMul()(self, other)
    
    #定义计算函数

    def matmul(self, other):
        return mytorch.ops.MatMul()(self, other)

    def sum(self, axes=None):
        return mytorch.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return mytorch.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return mytorch.ops.Reshape(shape)(self)

    def __neg__(self):
        return mytorch.ops.Negate()(self)

    def transpose(self, axes=None):
        return mytorch.ops.Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__


def compute_gradient_of_variables(output_tensor, out_grad):
    """获取输出节点相对于 node_list 中每个节点的梯度。
    将计算结果存储在每个变量的 grad 字段中。
    """
    # 每个节点对于输出梯度的来贡献来源（一个点对应不同的路径）
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # 特别注意初始化梯度
    # 我们实际上是在求标量reduce_sum(output_node)的导数
    # 而不是向量output_node。 但这是损失函数的常见情况使用sum。
    node_to_output_grads_list[output_tensor] = [out_grad]

    # 反向节点序列
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    for node_i in reverse_topo_order:
        adjoint = node_to_output_grads_list[node_i]
        node_i.grad = sum(adjoint)
        if node_i.op is None:
            continue
        partial_vk_to_i_list = node_i.op.gradient_as_tuple(node_i.grad, node_i)
        for node_k, partial_vk_to_i in zip(node_i.inputs, partial_vk_to_i_list):
            node_to_output_grads_list.setdefault(node_k, list())
            node_to_output_grads_list[node_k].append(partial_vk_to_i)


def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """
        做个拓扑排序，画出计算图
    """
    visited = set()
    topo_order = []
    topo_sort_dfs(node_list[-1], visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """反向 DFS"""
    for pre_node in node.inputs:
        topo_sort_dfs(pre_node, visited, topo_order)
    if node not in visited:
        topo_order.append(node)
        visited.add(node)
