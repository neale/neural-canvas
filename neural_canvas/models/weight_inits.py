import torch.nn as nn
import numpy as np
import warnings


def init_weights_normal(module, means, stds):
    if len(list(module.modules())) == 0:
        warnings.warn('No modules found to init. Returning module unchanged.')
        return module
    if not hasattr(means, '__iter__'):
        means = [means] * len(list(module.modules()))
    if not hasattr(stds, '__iter__'):
        stds = [stds] * len(list(module.modules()))
    for i, layer in enumerate(module.modules()):
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight.data,
                            means[i],
                            stds[i])
            if hasattr(layer.bias, 'data'):
                nn.init.constant_(layer.bias.data, 0.0)
        elif isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight.data,
                            means[i],
                            stds[i])
            if hasattr(layer.bias, 'data'):
                nn.init.constant_(layer.bias.data, 0.0)

    return module


def init_weights_uniform(module, mins, maxs):
    if len(list(module.modules())) == 0:
        warnings.warn('No modules found to init. Returning module unchanged.')
        return module
    if not hasattr(mins, '__iter__'):
        mins = [mins] * len(list(module.modules()))
    if not hasattr(maxs, '__iter__'):
        maxs = [maxs] * len(list(module.modules()))
    for i, layer in enumerate(module.modules()):
        if isinstance(layer, nn.Linear):
            nn.init.uniform_(layer.weight.data,
                             mins[i],
                             maxs[i])
            if hasattr(layer.bias, 'data'):
                nn.init.constant_(layer.bias.data, 0.0)
        elif isinstance(layer, nn.Conv2d):
            nn.init.uniform_(layer.weight.data,
                             mins[i],
                             maxs[i])
            if hasattr(layer.bias, 'data'):
                nn.init.constant_(layer.bias.data, 0.0)
    return module


def init_weights_dip(module):
    if len(list(module.modules())) == 0:
        warnings.warn('No modules found to init. Returning module unchanged.')
        return module
    for i, layer in enumerate(module.modules()):
        if isinstance(layer, nn.Conv2d):
            if layer.out_channels == 3 or layer.out_channels == 1:
                nn.init.zeros_(layer.weight.data)   
            else:
                nn.init.normal_(layer.weight.data, 0, np.sqrt(1./layer.in_channels))
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight.data, 0, np.sqrt(1./layer.in_features))
            if hasattr(layer.bias, 'data'):
                nn.init.constant_(layer.bias.data, 0.0)

    return module


def init_weights_siren(module, omega_0=30., hidden_omega_0=30.):
    if len(list(module.modules())) == 0:
        warnings.warn('No modules found to init. Returning module unchanged.')
        return module
    for i, layer in enumerate(module.modules()):
        if i == 0:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight.data, -1/layer.in_features, 1/layer.in_features)
                if hasattr(layer.bias, 'data'):
                    nn.init.constant_(layer.bias.data, 0.0)
        elif i == len(list(module.modules())) - 1:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight.data,
                                 -np.sqrt(6./layer.in_features) / hidden_omega_0,
                                 np.sqrt(6./layer.in_features) / hidden_omega_0)
                if hasattr(layer.bias, 'data'):
                    nn.init.constant_(layer.bias.data, 0.0)  
        else:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight.data,
                                 -np.sqrt(6./layer.in_features) / omega_0,
                                 np.sqrt(6./layer.in_features) / omega_0)
                if hasattr(layer.bias, 'data'):
                    nn.init.constant_(layer.bias.data, 0.0)
            elif isinstance(layer, nn.Conv2d):
                nn.init.uniform_(layer.weight.data,
                                 -np.sqrt(6./layer.in_channels) / omega_0,
                                 np.sqrt(6./layer.in_channels) / omega_0)
                if hasattr(layer.bias, 'data'):
                    nn.init.constant_(layer.bias.data, 0.0)

    return module

