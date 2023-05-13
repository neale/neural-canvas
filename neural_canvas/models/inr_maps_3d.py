import torch
import torch.nn as nn
import networkx

from neural_canvas.models.ops import *
from neural_canvas.models.torchgraph import randact, build_random_graph, TorchGraph


class INRRandomGraph3D(nn.Module):
    def __init__(self,
                 latent_dim,
                 c_dim,
                 layer_width,
                 num_graph_nodes,
                 graph=None,
                 activations='fixed',
                 final_activation='sigmoid',
                 name='INRRandomGraph3D'):
        super(INRRandomGraph3D, self).__init__()
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.layer_width = layer_width
        self.input_nodes = 1
        self.nodes = num_graph_nodes
        self.activations = activations
        self.final_activation = final_activation
        
        self.name = name
        self.linear_latents = nn.Linear(self.latent_dim, self.layer_width)
        self.linear_x = nn.Linear(1, self.layer_width, bias=False)
        self.linear_y = nn.Linear(1, self.layer_width, bias=False)
        self.linear_z = nn.Linear(1, self.layer_width, bias=False)
        self.linear_r = nn.Linear(1, self.layer_width, bias=False)
        self.linear1 = nn.Linear(self.layer_width, self.layer_width)
        self.linear2 = nn.Linear(self.layer_width, self.layer_width)
        self.scale = ScaleAct()
        if self.activations == 'random':
            acts = [randact(activation_set='large') for _ in range(7)]
        elif self.activations == 'fixed':
            acts = [nn.SiLU(), nn.ELU(), nn.Softplus(), nn.Tanh(), 
                    Gaussian(), SinLayer(), nn.GELU()]
        elif hasattr(torch.nn, activations):
            acts = [getattr(torch.nn, activations)() for _ in range(7)]
        else:
            raise ValueError('activations must be fixed, random, ' \
                             f'or else a valid torch.nn activation, got `{activations}`')
        self.acts_start = nn.ModuleList(acts)
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
                                  activation_set='large')
        if final_activation == 'sigmoid':
            self.act_out = nn.Sigmoid()
        elif final_activation == 'tanh':
            self.act_out = nn.Tanh()
        elif final_activation is None:
            self.act_out = nn.Identity()

    def generate_act_list(self):
        acts = [randact(activation_set='large') for _ in range(7)]
        self.acts = nn.ModuleList(acts)

    def get_graph_str(self):
        s = ''.join(networkx.generate_graphml(self.graph))
        return s

    def load_graph_str(self, s):
        return networkx.parse_graphml(s, node_type=int)

    def forward(self, inputs, latents=None):
        x, y, z, r = inputs[0, 0, ...], inputs[0, 1, ...], inputs[0, 2, ...], inputs[0, 3, ...]
        z_ = self.acts_start[1](self.linear_z(z))
        r_ = self.acts_start[2](self.linear_r(r))
        y_ = self.acts_start[3](self.linear_y(y))
        x_ = self.acts_start[4](self.linear_x(x))
        f = self.acts_start[5](z_ + x_ + y_ + r_)
        if latents is not None:
            latents_ = self.acts_start[0](self.linear_latents(latents))
            f = f + latents_
        f = self.acts_start[6](self.linear1(f))
        res = self.scale(self.network(f))
        res = self.act_out(res)
        return res


class INRLinearMap3D(nn.Module):
    def __init__(self,
                 latent_dim,
                 c_dim,
                 layer_width,
                 activations='fixed',
                 final_activation='sigmoid',
                 name='INRLinearMap3D'):
        super(INRLinearMap3D, self).__init__()
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.layer_width = layer_width
        self.activations = activations
        self.final_activation = final_activation

        self.name = name
        
        self.linear_latents = nn.Linear(self.latent_dim, self.layer_width)
        self.linear_x = nn.Linear(1, self.layer_width, bias=False)
        self.linear_y = nn.Linear(1, self.layer_width, bias=False)
        self.linear_z = nn.Linear(1, self.layer_width, bias=False)
        self.linear_r = nn.Linear(1, self.layer_width, bias=False)

        self.linear1 = nn.Linear(self.layer_width, self.layer_width)
        self.linear2 = nn.Linear(self.layer_width, self.layer_width)
        self.linear3 = nn.Linear(self.layer_width, self.layer_width)
        self.linear4 = nn.Linear(self.layer_width, self.c_dim)

        if self.activations == 'random':
            acts = [randact(activation_set='large') for _ in range(5)]
        elif self.activations == 'fixed':
            acts = [nn.GELU(), nn.Tanh(), nn.ELU(), SinLayer(), ScaleAct()]
        elif hasattr(torch.nn, activations):
            acts = [getattr(torch.nn, activations)() for _ in range(5)]
        else:
            raise ValueError('activations must be `fixed`, `random`, ' \
                             f'or else a valid torch.nn activation, got `{activations}`')
        self.acts = nn.ModuleList(acts)

        if self.final_activation == 'sigmoid':
            self.act_out = torch.sigmoid
        elif self.final_activation == 'tanh':
            self.act_out = torch.tanh
        elif final_activation is None:
            self.act_out = nn.Identity()

    def generate_new_acts(self):
        acts = [randact(activation_set='large') for _ in range(5)]
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
        latents_pt = self.linear_latents(latents)
        x_pt = self.linear_x(x)
        y_pt = self.linear_y(y)
        r_pt = self.linear_r(r)
        z_pt = self.linear_z(z)
        z = z_pt + x_pt + y_pt + r_pt
        if latents is not None:
            latents_pt = self.linear_latents(latents)
            z = z + latents_pt
        z = self.acts[0](z)
        z = self.acts[1](self.linear1(z))
        z = self.acts[2](self.linear2(z))
        z = self.acts[3](self.linear3(z))
        x = self.acts[4](self.linear4(z))
        x = self.act_out(x)
        return x



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