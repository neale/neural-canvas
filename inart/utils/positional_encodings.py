import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats
from sklearn.datasets import make_blobs


def coordinates_2D(x_dim, y_dim, batch_size, zoom, pan, scale, device, as_mat=False):
    xpan, ypan = pan
    xzoom, yzoom = zoom
    n_pts = x_dim * y_dim
    
    x_range = scale * (torch.arange(x_dim) - (x_dim - 1) / xpan) / (x_dim - 1) / xzoom
    y_range = scale * (torch.arange(y_dim) - (y_dim - 1) / ypan) / (y_dim - 1) / yzoom
    
    x_mat, y_mat = torch.meshgrid(x_range, y_range)
    r_mat = torch.sqrt(x_mat * x_mat + y_mat * y_mat)
    if as_mat:
        return x_mat, y_mat, r_mat, n_pts
    x_vec = x_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
    y_vec = y_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
    r_vec = r_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
    
    x_vec = x_vec.float().to(device)
    y_vec = y_vec.float().to(device)
    r_vec = r_vec.float().to(device)
    
    return x_vec, y_vec, r_vec, n_pts


def coordinates_3D(x_dim, y_dim, z_dim, batch_size, zoom, pan, scale, device, as_mat=False):
    xpan, ypan, zpan = pan
    xzoom, yzoom, zzoom = zoom
    n_pts = x_dim * y_dim * z_dim
    
    x_range = scale * (torch.arange(x_dim) - (x_dim - 1) / xpan) / (x_dim - 1) / xzoom
    y_range = scale * (torch.arange(y_dim) - (y_dim - 1) / ypan) / (y_dim - 1) / yzoom
    z_range = scale * (torch.arange(z_dim) - (z_dim - 1) / zpan) / (z_dim - 1) / zzoom
    
    x_mat, y_mat, z_mat = torch.meshgrid(x_range, y_range, z_range)
    r_mat = torch.sqrt(x_mat * x_mat + y_mat * y_mat + z_mat * z_mat)
    if as_mat:
        return x_mat, y_mat, r_mat, n_pts
    x_vec = x_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
    y_vec = y_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
    z_vec = z_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
    r_vec = r_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
    
    x_vec = x_vec.float().to(device)
    y_vec = y_vec.float().to(device)
    z_vec = z_vec.float().to(device)
    r_vec = r_vec.float().to(device)
    
    return x_vec, y_vec, z_vec, r_vec, n_pts


def mixture_2D(x_dim, y_dim, device):
    n_components = 3
    X, truth = make_blobs(
        n_samples=100,
        centers=n_components, 
        cluster_std=np.random.randint(1, 5, size=(n_components,)))
    x = X[:, 0]
    y = X[:, 1]# Define the borders
    deltaX = (max(x) - min(x))/10
    deltaY = (max(y) - min(y))/10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    xx, yy = np.mgrid[xmin:xmax:complex(0,x_dim), ymin:ymax:complex(0,y_dim)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    xy_mat = np.reshape(kernel(positions).T, xx.shape)
    xy_mat = torch.from_numpy(xy_mat).float().to(device)
    return xy_mat


class FourierEncoding(nn.Module):
    # instatiate for all components of the input parameterizations
    # E.g. for a 2D parameterization, instantiate 2 FourierEncoding modules for x and y
    def __init__(self, num_layers):
        super().__init__()
        frequencies = [2 ** i for i in range(num_layers)]
        self.register_buffer('freqs', torch.Tensor(frequencies)[None, :, None])

    def forward(self, x):
        x = x * self.freqs
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        return x