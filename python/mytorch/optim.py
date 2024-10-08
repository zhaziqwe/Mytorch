from mytorch.autograd import Tensor
from typing import List

class Optimizer:

    def __init__(self, params:List[Tensor]):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):

    def __init__(self, params:List[Tensor], lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params: 
            grad = self.u.get(param, 0) * self.momentum + (1 - self.momentum) * (param.grad.cached_data+ self.weight_decay * param.cached_data)
            self.u[param] = grad 
            param.cached_data -= self.lr * grad 


class Adam(Optimizer):

    def __init__(
        self,
        params:List[Tensor],
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for param in self.params: 
            grad_with_wd = param.grad.cached_data + self.weight_decay * param.cached_data
            
            new_m = self.m.get(param, 0) * self.beta1 + (1 - self.beta1) * grad_with_wd
            new_v = self.v.get(param, 0) * self.beta2 + (1 - self.beta2) * grad_with_wd ** 2
            self.m[param] = new_m
            self.v[param] = new_v
            m_with_bias = (new_m / (1 - self.beta1 ** self.t))
            v_with_bias = (new_v / (1 - self.beta2 ** self.t))

            update = self.lr * (m_with_bias) / (v_with_bias ** 0.5 + self.eps)
            param.cached_data -= update