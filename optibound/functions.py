import torch
import numpy as np 
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class BaseFunction(ABC):

    def __init__(self, dimension):

        self.dimension = dimension

    @abstractmethod
    def __call__(self, x):

        pass


    def gradient(self, x):

        x_clone = x.clone().detach().requires_grad_(True)

        val = self(x_clone)

        grad = torch.autograd.grad(val.sum(), x_clone)[0]

        return grad

class EllipticBowl(BaseFunction):

    def __init__(self, a=0.5, b=5):

        super().__init__(dimension=2)

        self.a = a
        self.b = b


    def __call__(self, x):

        return self.a * x[..., 0]**2 + self.b * x[..., 1]**2


