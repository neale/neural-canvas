import torch.nn as nn
import numpy as np


def init_weights_normal(module, means, stds):
    if not hasattr(means, '__iter__'):
        mins = [mins] * len(list(module.modules()))
    if not hasattr(stds, '__iter__'):
        maxs = [maxs] * len(list(module.modules()))
    for i, layer in enumerate(module.modules()):
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight.data,
                            means[i],
                            stds[i])
            nn.init.constant_(layer.bias.data, 0.0)
        elif isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight.data,
                            means[i],
                            stds[i])
            nn.init.constant_(layer.bias.data, 0.0)

    return module


def init_weights_uniform(module, mins, maxs):
    if not hasattr(mins, '__iter__'):
        mins = [mins] * len(list(module.modules()))
    if not hasattr(maxs, '__iter__'):
        maxs = [maxs] * len(list(module.modules()))
    for i, layer in enumerate(module.modules()):
        if isinstance(layer, nn.Linear):
            nn.init.uniform_(layer.weight.data,
                             mins[i],
                             maxs[i])
            nn.init.constant_(layer.bias.data, 0.0)
        elif isinstance(layer, nn.Conv2d):
            nn.init.uniform_(layer.weight.data,
                             mins[i],
                             maxs[i])
            nn.init.constant_(layer.bias.data, 0.0)
    return module


def init_weights_dip(module):
    for i, layer in enumerate(module.modules()):
        if isinstance(layer, nn.Conv2d):
            if layer.out_channels == 3:
                nn.init.zeros_(layer.weight.data)   
            else:
                nn.init.normal_(layer.weight.data, 0, np.sqrt(1./layer.in_channels))
            
            nn.init.constant_(layer.bias.data, 0.0)

    return module


def init_weights_siren(module, omega_0=30., hidden_omega_0=30.):
    for i, layer in enumerate(module.modules()):
        if i == 0:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight.data, -1/layer.in_features, 1/layer.in_features)
                nn.init.constant_(layer.bias.data, 0.0)
        elif i == len(list(module.modules())) - 1:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight.data,
                                 -np.sqrt(6./layer.in_features) / hidden_omega_0,
                                 np.sqrt(6./layer.in_features) / hidden_omega_0)
                nn.init.constant_(layer.bias.data, 0.0)    
        else:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight.data,
                                 -np.sqrt(6./layer.in_features) / omega_0,
                                 np.sqrt(6./layer.in_features) / omega_0)
                nn.init.constant_(layer.bias.data, 0.0)
            elif isinstance(layer, nn.Conv2d):
                nn.init.uniform_(layer.weight.data,
                                 -np.sqrt(6./layer.in_channels) / omega_0,
                                 np.sqrt(6./layer.in_channels) / omega_0)
                nn.init.constant_(layer.bias.data, 0.0)

    return module

