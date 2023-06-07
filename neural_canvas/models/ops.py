import torch
import torch.nn as nn
import torch.autograd as autograd


class Gaussian(nn.Module):
    def __init__(self):
        super(Gaussian, self).__init__()

    def forward(self, x, a=1.0):
        return a * torch.exp((-x ** 2) / (2 * a ** 2)) 


class SinLayer(nn.Module):
    def __init__(self, scale=1.0):
        super(SinLayer, self).__init__()
        self.scale = scale

    def forward(self, x):
        return torch.sin(self.scale * x)


class CosLayer(nn.Module):
    def __init__(self):
        super(CosLayer, self).__init__()

    def forward(self, x):
        return torch.cos(x)


class ScaleAct(nn.Module):
    def __init__(self):
        super(ScaleAct, self).__init__()
        r = torch.ones(1,).uniform_(1, 10)
        self.r = nn.Parameter(r)

    def forward(self, x):
        return x * self.r


class AddAct(nn.Module):
    def __init__(self):
        super(AddAct, self).__init__()
        r = torch.ones(1,).uniform_(1, 10)
        self.r = nn.Parameter(r)

    def forward(self, x):
        return x + self.r
    

class STEFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return torch.nn.functional.hardtanh(grad_output)  # clamp grads


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x
    