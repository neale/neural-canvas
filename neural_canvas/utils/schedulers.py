""" Schedulers for iterating through a range of values """
import torch
import math
import numpy as np


def linear(start, end, steps):
    """Linear schedule from start to end in steps"""
    return torch.linspace(start, end, steps)


def geometric(start, end, steps):
    """Geometric schedule from start to end in steps"""
    return torch.from_numpy(np.geomspace(start, end, steps))


def cosine(start, end, steps):
    """Cosine schedule from start to end in steps"""
    return (end - start) * 0.5 * (1 + torch.cos(torch.linspace(0, math.pi, steps))) + start


def sigmoid(start, end, steps):
    """Sigmoid schedule from start to end in steps"""
    return (end - start) * torch.sigmoid(torch.linspace(-5, 5, steps)) + start


def exp(start, end, steps):
    """Exponential schedule from start to end in steps"""
    return (end - start) * torch.exp(torch.linspace(-5, 5, steps)) + start


def log(start, end, steps):
    """Logarithmic schedule from start to end in steps"""
    return (end - start) * torch.log(torch.linspace(0.1, 10, steps)) + start


def sqrt(start, end, steps):
    """Sublinear schedule from start to end in steps"""
    return (end - start) * torch.sqrt(torch.linspace(0, 1, steps)) + start


def cbrt(start, end, steps):
    """Cubic root schedule from start to end in steps"""
    return (end - start) * torch.cbrt(torch.linspace(0, 1, steps)) + start