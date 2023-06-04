import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats


def coordinates_2D(x_dim, y_dim, batch_size, zoom, pan, scale, as_mat=False):
    xpan, ypan = pan
    xzoom, yzoom = zoom
    n_pts = x_dim * y_dim
    
    x_range = scale * (torch.arange(x_dim) - (x_dim - 1) / xpan) / (x_dim - 1) / xzoom
    y_range = scale * (torch.arange(y_dim) - (y_dim - 1) / ypan) / (y_dim - 1) / yzoom
    
    x_mat, y_mat = torch.meshgrid(x_range, y_range, indexing='ij')
    r_mat = torch.sqrt(x_mat * x_mat + y_mat * y_mat)
    if as_mat:
        return x_mat, y_mat, r_mat
    x_vec = x_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
    y_vec = y_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
    r_vec = r_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
    
    return x_vec, y_vec, r_vec


def coordinates_3D(x_dim, y_dim, z_dim, batch_size, zoom, pan, scale, as_mat=False):
    xpan, ypan, zpan = pan
    xzoom, yzoom, zzoom = zoom
    n_pts = x_dim * y_dim * z_dim
    
    x_range = scale * (torch.arange(x_dim) - (x_dim - 1) / xpan) / (x_dim - 1) / xzoom
    y_range = scale * (torch.arange(y_dim) - (y_dim - 1) / ypan) / (y_dim - 1) / yzoom
    z_range = scale * (torch.arange(z_dim) - (z_dim - 1) / zpan) / (z_dim - 1) / zzoom
    
    x_mat, y_mat, z_mat = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
    r_mat = torch.sqrt(x_mat * x_mat + y_mat * y_mat + z_mat * z_mat)
    if as_mat:
        return x_mat, y_mat, z_mat, r_mat
    x_vec = x_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
    y_vec = y_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
    z_vec = z_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
    r_vec = r_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
    
    return x_vec, y_vec, z_vec, r_vec


class FourierEncoding(nn.Module):
    # instatiate for all components of the input parameterizations
    def __init__(self, num_freqs):
        super().__init__()
        frequencies = [2 ** i * np.pi for i in range(num_freqs)]
        self.register_buffer('freqs', torch.Tensor(frequencies)[None, :, None, None])

    def forward(self, x):
        freq_outs = []
        for i in range(x.shape[1]):
            f = self.freqs * x[:, i]
            freq_outs.append(torch.cat([f.sin(), f.cos()], dim=1))
        encoded_input = torch.cat(freq_outs, dim=1)
        return encoded_input
