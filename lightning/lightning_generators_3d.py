import torch
import torch.nn as nn
import networkx

from neural_canvas.models.ops import (
    ScaleAct,
    Gaussian,
    SinLayer,
    CosLayer
)
from neural_canvas.models.torchgraph import randact, build_random_graph, TorchGraph


class StraightBackbone(nn.Module):
    def __init__(self, num_layers, layer_width, c_dim):
        super(StraightBackbone, self).__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.layers = nn.ModuleList([nn.Linear(layer_width, layer_width) for _ in range(num_layers)])
        self.layer_out = nn.Linear(layer_width, c_dim) if c_dim else None


    def forward(self, x, acts):
        for layer, act in zip(self.layers, acts):
            x = act(layer(x))
        if self.layer_out:
            x = acts[-1](self.layer_out(x))
        return x


class ResNetBackbone(nn.Module):
    # post act resnet
    def __init__(self, num_layers, layer_width, c_dim):
        super(ResNetBackbone, self).__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.layers = nn.ModuleList([nn.Linear(layer_width, layer_width) for _ in range(num_layers)])
        self.layer_out = nn.Linear(layer_width, c_dim) if c_dim else None


    def forward(self, x, acts):
        for layer, act in zip(self.layers, acts):
            x = act(layer(x) + x)
        if self.layer_out:
            x = acts[-1](self.layer_out(x))
        return x
    

class DenseNetBackbone(nn.Module):
    def __init__(self, num_layers, layer_width, c_dim):
        super(DenseNetBackbone, self).__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width         
        self.layers = nn.ModuleList([])   
        for i in range(self.num_layers):
            self.layers.append(nn.Linear(self.layer_width*(i+1), self.layer_width))
        self.layer_out = nn.Linear(self.layer_width*(i+2), c_dim) if c_dim else None
    
    def forward(self, x, acts):
        for layer, act in zip(self.layers, acts):
            x = torch.cat([x, act(layer(x))], -1)
        if self.layer_out:
            x = acts[-1](self.layer_out(x))
        return x
    

class UNetBackbone(nn.Module):
    def __init__(self, num_layers, layer_width, c_dim):
        super(UNetBackbone, self).__init__()
        self.num_layers = num_layers
        assert self.num_layers % 2 == 0, f'num_layers must be even for Unet backbone. Got {self.num_layers}'
        self.layer_width = layer_width
        self.layers = nn.ModuleList([nn.Linear(layer_width, layer_width) for _ in range(num_layers//2+1)])
        for _ in range(self.num_layers//2-1):
            self.layers.append(nn.Linear(layer_width*2, layer_width))
        self.layer_out = nn.Linear(layer_width, c_dim) if c_dim else None
        
    def forward(self, x, acts):
        # down pass
        down = []
        for i in range(self.num_layers//2):
            x = acts[i](self.layers[i](x))
            down.append(x)
        # up pass
        for k in range(1, self.num_layers//2):
            if k > 1:
                x = torch.cat([x, down[-k]], -1)
            x = acts[k+i](self.layers[k+i](x))
        if self.layer_out:
            x = acts[-1](self.layer_out(x))
        return x
    

class INRRandomGraph3D(nn.Module):
    def __init__(self,
                 latent_dim,
                 c_dim,
                 layer_width,
                 input_encoding_dim,
                 num_graph_nodes,
                 graph=None,
                 activations='fixed',
                 final_activation='sigmoid',
                 backbone='straight',
                 num_layers=3,
                 device='cpu',
                 name='INRRandomGraph3D'):
        super(INRRandomGraph3D, self).__init__()
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.layer_width = layer_width
        self.input_encoding_dim = input_encoding_dim
        
        self.input_nodes = 1
        self.nodes = num_graph_nodes
        self.activations = activations
        self.num_layers = num_layers
        self.final_activation = final_activation
        self.device = device

        self.name = name
        
        self.linear_latents = nn.Linear(self.latent_dim, self.layer_width)
        self.linear_x = nn.Linear(self.input_encoding_dim, self.layer_width, bias=False)
        self.linear_y = nn.Linear(self.input_encoding_dim, self.layer_width, bias=False)
        self.linear_z = nn.Linear(self.input_encoding_dim, self.layer_width, bias=False)
        self.linear_r = nn.Linear(self.input_encoding_dim, self.layer_width, bias=False)
        self.scale = ScaleAct(device)
            
        if backbone == 'straight':
            self.backbone = StraightBackbone(self.num_layers, self.layer_width, None)
        elif backbone == 'resnet':
            self.backbone = ResNetBackbone(self.num_layers, self.layer_width, None)
        elif backbone == 'densenet':
            self.backbone = DenseNetBackbone(self.num_layers, self.layer_width, None)
        elif backbone == 'unet':
            self.backbone = UNetBackbone(self.num_layers, self.layer_width, None)
        else:
            raise ValueError(f'backbone must be `straight`, `resnet`, `densenet`, `unet`: got {backbone}')
    
        if self.activations == 'random':
            acts = self.generate_new_acts()
        elif self.activations == 'fixed':
            self.acts = nn.ModuleList([nn.SiLU(), nn.ELU(), nn.Softplus(), nn.Tanh(), 
                    Gaussian(), SinLayer(), nn.GELU()])
        elif hasattr(torch.nn, activations):
            self.acts = nn.ModuleList([getattr(torch.nn, activations)() for _ in range(5+num_layers)])
        else:
            raise ValueError('activations must be fixed, random, ' \
                             f'or else a valid torch.nn activation, got `{activations}`')

        # init graph parameters
        k = 4
        p = .75
        out_nodes = 1
        combine = False
        if out_nodes == 1:
            combine=True
        if graph is None:
            self.graph = build_random_graph(
                self.nodes,
                self.input_nodes,
                out_nodes,
                p,
                k)
        else:
            self.graph = self.load_graph_str(graph)

        self.network = TorchGraph(self.graph,
                                  self.layer_width,
                                  self.layer_width,
                                  self.c_dim,
                                  combine,
                                  activation_set='large',
                                  device=device)

        if final_activation == 'sigmoid':
            self.act_out = nn.Sigmoid()
        elif final_activation == 'tanh':
            self.act_out = nn.Tanh()
        elif final_activation is None:
            self.act_out = nn.Identity()

    def generate_new_acts(self):
        acts = nn.ModuleList([])
        for i in range(6+self.num_layers):
            if i < 5:
                acts.append(randact(activation_set='start', device=self.device))
            else:
                acts.append(randact(activation_set='large', device=self.device))
        self.acts = nn.ModuleList(acts)    

    def get_graph_str(self):
        s = ''.join(networkx.generate_graphml(self.graph))
        return s

    def load_graph_str(self, s):
        return networkx.parse_graphml(s, node_type=int)
    
    def forward(self, fields, latents=None):
        if fields.ndim == 4: # after positional encoding
            chunk_size = fields.shape[1]//4
            x = fields[:, :chunk_size, :, 0].permute(0, 2, 1)
            y = fields[:, chunk_size:2*chunk_size, :, 0].permute(0, 2, 1)
            z = fields[:, 2*chunk_size:3*chunk_size, :, 0].permute(0, 2, 1)
            r = fields[:, chunk_size*3:, :, 0].permute(0, 2, 1)
        else:
            x, y, z, r = fields[:, 0, ...], fields[:, 1, ...], fields[:, 2, ...], fields[:, 3, ...]
        x_pt = self.acts[0](self.linear_x(x))
        y_pt = self.acts[1](self.linear_y(y))
        z_pt = self.acts[2](self.linear_z(z))
        r_pt = self.acts[3](self.linear_r(r))
        f = x_pt + y_pt + z_pt + r_pt
        if latents is not None:
            f += self.acts[4](self.linear_latents(latents))
        f = self.backbone(f, self.acts[5:-1])
        f_out = self.acts[-1](self.scale(self.network(f)))
        return f_out
    


class INRLinearMap3D(nn.Module):
    def __init__(self,
                 latent_dim,
                 c_dim,
                 layer_width,
                 input_encoding_dim,
                 activations='fixed',
                 final_activation='sigmoid',
                 backbone='straight',
                 num_layers=3,
                 device='cpu',
                 name='INRLinearMap3D'):
        super(INRLinearMap3D, self).__init__()
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.layer_width = layer_width
        self.input_encoding_dim = input_encoding_dim
        self.activations = activations
        self.num_layers = num_layers
        self.final_activation = final_activation
        self.device = device

        self.name = name
        
        self.linear_latents = nn.Linear(self.latent_dim, self.layer_width)
        self.linear_x = nn.Linear(self.input_encoding_dim, self.layer_width, bias=False)
        self.linear_y = nn.Linear(self.input_encoding_dim, self.layer_width, bias=False)
        self.linear_z = nn.Linear(self.input_encoding_dim, self.layer_width, bias=False)
        self.linear_r = nn.Linear(self.input_encoding_dim, self.layer_width, bias=False)
        
        if backbone == 'straight':
            self.backbone = StraightBackbone(self.num_layers, self.layer_width, self.c_dim)
        elif backbone == 'resnet':
            self.backbone = ResNetBackbone(self.num_layers, self.layer_width, self.c_dim)
        elif backbone == 'densenet':
            self.backbone = DenseNetBackbone(self.num_layers, self.layer_width, self.c_dim)
        elif backbone == 'unet':
            self.backbone = UNetBackbone(self.num_layers, self.layer_width, self.c_dim)
        else:
            raise ValueError(f'backbone must be `straight`, `resnet`, `densenet`, `unet`: got {backbone}')
        
        if self.activations == 'random':
            self.generate_new_acts()
        elif self.activations == 'fixed':
            self.acts = nn.ModuleList([nn.Tanhshrink(), nn.Tanh(), nn.ELU(), SinLayer(), 
                    nn.RReLU(), nn.ELU(), nn.SiLU(), SinLayer(), ScaleAct(device)])
        elif hasattr(torch.nn, activations):
            self.acts = nn.ModuleList([getattr(torch.nn, activations)() for _ in range(5+num_layers)])
        else:
            raise ValueError('activations must be `fixed`, `random`, ' \
                             f'or else a valid torch.nn activation, got `{activations}`')

        if self.final_activation == 'sigmoid':
            self.act_out = torch.sigmoid
        elif self.final_activation == 'tanh':
            self.act_out = torch.tanh
        elif final_activation is None:
            self.act_out = nn.Identity()

    def generate_new_acts(self):
        acts = nn.ModuleList([])
        for i in range(5+self.num_layers):
            if i < 5:
                acts.append(randact(activation_set='start', device=self.device))
            else:
                acts.append(randact(activation_set='large', device=self.device))
        self.acts = nn.ModuleList(acts)    

    def get_graph(self):
        g = networkx.Graph()
        for i in range(6):
            g.add_node(i)
        for i in range(4):
            g.add_edge(i, 4)
        g.add_edge(4, 4)
        g.add_edge(4, 4)
        g.add_edge(4, 4)
        g.add_edge(4, 5)

        for node in range(6):
            g.nodes[node]['forcelabels'] = False
            g.nodes[node]['shape'] = 'circle'
            g.nodes[node]['id'] = ''
            g.nodes[node]['label'] = ''
            g.nodes[node]['rotatation'] = 180
            g.nodes[node]['bgcolor'] = 'transparent'
        g.bgcolor = "transparent"

        return g
    
    def forward(self, inputs, latents=None):
        x, y, z, r = inputs[:, 0, ...], inputs[:, 1, ...], inputs[:, 2, ...], inputs[:, 3, ...]
        x_pt = self.acts[0](self.linear_x(x))
        y_pt = self.acts[1](self.linear_y(y))
        r_pt = self.acts[2](self.linear_r(r))
        z_pt = self.acts[3](self.linear_z(z))
        z = z_pt + x_pt + y_pt + r_pt
        if latents is not None:
            latents_pt = self.acts[4](self.linear_latents(latents))
            z = z + latents_pt
        z = self.acts[5](z)
        z = self.acts[6](self.linear1(z))
        z = self.acts[7](self.linear2(z))
        z = self.acts[8](self.linear3(z))
        z_out = self.act_out(self.linear4(z))
        return z_out

    def forward(self, fields, latents=None):
        #TODO refactor this to look better, its clunky to support positional encodings
        if fields.ndim == 4: # after positional encoding
            chunk_size = fields.shape[1]//4
            x = fields[:, :chunk_size, :, 0].permute(0, 2, 1)
            y = fields[:, chunk_size:2*chunk_size, :, 0].permute(0, 2, 1)
            z = fields[:, 2*chunk_size:3*chunk_size, :, 0].permute(0, 2, 1)
            r = fields[:, chunk_size*3:, :, 0].permute(0, 2, 1)
        else:
            x, y, z, r = fields[:, 0, ...], fields[:, 1, ...], fields[:, 2, ...], fields[:, 3, ...]
        x_pt = self.acts[0](self.linear_x(x))
        y_pt = self.acts[1](self.linear_y(y))
        z_pt = self.acts[2](self.linear_z(z))
        r_pt = self.acts[3](self.linear_r(r))
        f = x_pt + y_pt + z_pt + r_pt
        if latents is not None:
            f += self.acts[4](self.linear_latents(latents))
        f_out = self.backbone(f, self.acts[5:])
        return f_out


class INRConvMap3D(nn.Module):
    def __init__(self,
                 latent_dim,
                 c_dim,
                 feature_dim,
                 activations='fixed',
                 final_activation='sigmoid',
                 name='INRConvMap3D'):
        super(INRConvMap3D, self).__init__()
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.feature_dim = feature_dim
        self.activations = activations
        self.final_activation = final_activation

        self.name = name

        self.conv1 = nn.Conv3d(2, self.feature_dim, kernel_size=1, stride=1, padding='same')
        self.conv2 = nn.Conv3d(self.feature_dim, self.feature_dim, kernel_size=1, stride=1, padding='same')
        self.conv3 = nn.Conv3d(self.feature_dim, self.feature_dim, kernel_size=1, stride=1, padding='same')
        self.conv4 = nn.Conv3d(self.feature_dim, self.feature_dim, kernel_size=1, stride=1, padding='same')
        self.conv5 = nn.Conv3d(self.feature_dim, self.feature_dim, kernel_size=1, stride=1, padding='same')
        self.conv6 = nn.Conv3d(self.feature_dim, self.feature_dim, kernel_size=1, stride=1, padding='same')
        self.conv7 = nn.Conv3d(self.feature_dim, self.feature_dim, kernel_size=1, stride=1, padding='same')
        self.conv8 = nn.Conv3d(self.feature_dim, self.feature_dim, kernel_size=1, stride=1, padding='same')
        self.conv_rgb = nn.Conv3d(self.feature_dim, 3, kernel_size=1, stride=1, padding='same')

        if self.activations == 'random':
            acts = [randact(activation_set='large') for _ in range(8)]
        elif self.activations == 'fixed':
            acts = [nn.GELU(), nn.ELU(), nn.Softplus(), nn.Tanh(), SinLayer(),
                    nn.Tanh(), nn.ELU(), nn.Softplus(), CosLayer(), ScaleAct()]
        elif hasattr(torch.nn, activations):
            acts = [getattr(torch.nn, activations)() for _ in range(8)]
        else:
            raise ValueError('activations must be `fixed`, `random`, ' \
                             f'or else a valid torch.nn activation, got `{activations}`')
        self.acts = nn.ModuleList(acts)

        self.norms = nn.ModuleList([nn.Identity() for _ in range(8)])

        if final_activation == 'sigmoid':
            self.act_out = torch.sigmoid
        elif final_activation == 'tanh':
            self.act_out = torch.tanh

    def generate_new_acts(self):
        acts = [randact(activation_set='large') for _ in range(8)]
        self.acts = nn.ModuleList(acts)    
    
    def forward(self, inputs, latents=None):
        raise NotImplementedError
        if latents is not None:
            x = torch.cat([inputs, latents], 0).unsqueeze(0)
        x = self.acts[0](self.norms[0](self.conv1(x)))
        x = self.acts[1](self.norms[1](self.conv2(x)))
        x = self.acts[2](self.norms[2](self.conv3(x)))
        x = self.acts[3](self.norms[3](self.conv4(x)))
        x = self.acts[4](self.norms[4](self.conv5(x)))
        x = self.acts[5](self.norms[5](self.conv6(x)))
        x = self.acts[6](self.norms[6](self.conv7(x)))
        x = self.acts[7](self.norms[7](self.conv8(x)))
        x = self.act_out(self.conv_rgb(x))
        return x

if __name__ == '__main__':
    import time
    import numpy as np
    times = []
    for _ in range(20):
        s = time.time()
        model = INRRandomGraph3D()
        e = time.time()
        times.append(e - s)
    print('mean time to load default WS 3D Graph: ', np.mean(times))
    print(model)
