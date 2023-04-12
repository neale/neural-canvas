import torch
import torch.nn as nn
import networkx

from ops import *
from torchgraph import randact, build_random_graph, TorchGraph


class INRRandomGraph3D(nn.Module):
    def __init__(self,
                 noise_dim,
                 c_dim,
                 layer_width,
                 nodes,
                 graph=None,
                 activations='random'):
        super(INRRandomGraph3D, self).__init__()
        self.noise_dim = noise_dim
        self.c_dim = c_dim
        self.layer_width = layer_width
        self.input_nodes = 1
        self.nodes = nodes
        self.name = 'INRRandomGraph3D'

        self.linear_noise = nn.Linear(self.noise_dim, self.layer_width)
        self.linear_x = nn.Linear(1, self.layer_width, bias=False)
        self.linear_y = nn.Linear(1, self.layer_width, bias=False)
        self.linear_z = nn.Linear(1, self.layer_width, bias=False)
        self.linear_r = nn.Linear(1, self.layer_width, bias=False)
        self.linear1 = nn.Linear(self.layer_width, self.layer_width)
        if self.activations == 'random':
            acts = [randact(activation_set='large') for _ in range(8)]
        elif self.activations == 'basic':
            acts = [nn.Tanh(), nn.ELU(), nn.Softplus(), nn.Tanh(), 
                    Gaussian(), SinLayer()]
        elif hasattr(torch.nn, activations):
            acts = [getattr(torch.nn, activations)() for _ in range(8)]
        else:
            raise ValueError('activations must be basic, random, '\
                             'or else a valid torch.nn activation')
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
            print ('loading old graph')
            self.graph = self.load_graph_str(graph)

        self.network = TorchGraph(self.graph,
                                  self.layer_width,
                                  self.layer_width,
                                  self.c_dim,
                                  combine,
                                  activation_set='large')
        if self.clip_loss:
            self.act_out = torch.tanh
        else:
            self.act_out = torch.sigmoid

    def generate_act_list(self):
        acts = [randact(activation_set='large') for _ in range(9)]
        self.acts = nn.ModuleList(acts)

    def get_graph_str(self):
        s = ''.join(networkx.generate_graphml(self.graph))
        return s

    def load_graph_str(self, s):
        return networkx.parse_graphml(s, node_type=int)

    def forward(self, x, y, z, r, noise):
        noise_ = self.acts_start[0](self.linear_noise(noise))
        z_ = self.acts_start[1](self.linear_z(z))
        r_ = self.acts_start[2](self.linear_r(r))
        y_ = self.acts_start[3](self.linear_y(y))
        x_ = self.acts_start[4](self.linear_x(x))
        f = self.acts_start[5](z_ + x_ + y_ + r_ + noise_)
        f = self.acts_start[6](self.linear1(f))
        res = self.network(f)
        res = self.act_out(res)
        return res


class INRLinearMap3D(nn.Module):
    def __init__(self,
                 noise_dim,
                 c_dim,
                 layer_width,
                 activations='basic',
                 clip_loss=False,
                 name='INRLinearMap3D'):
        super(INRLinearMap3D, self).__init__()
        self.noise_dim = noise_dim
        self.c_dim = c_dim
        self.layer_width = layer_width
        self.name = name
        self.linear_noise = nn.Linear(self.z_dim, self.layer_width)
        self.linear_x = nn.Linear(1, self.layer_width, bias=False)
        self.linear_y = nn.Linear(1, self.layer_width, bias=False)
        self.linear_z = nn.Linear(1, self.layer_width, bias=False)
        self.linear_r = nn.Linear(1, self.layer_width, bias=False)

        self.linear1 = nn.Linear(self.layer_width, self.layer_width)
        self.linear2 = nn.Linear(self.layer_width, self.layer_width)
        self.linear3 = nn.Linear(self.layer_width, self.layer_width)
        self.linear4 = nn.Linear(self.layer_width, self.c_dim)

        if self.activations == 'random':
            acts = [randact(activation_set='large') for _ in range(9)]
        elif self.activations == 'basic':
            acts = [nn.Tanh(), nn.ELU(), nn.Softplus(), nn.Tanh(), SinLayer(),
                    nn.Tanh(), nn.ELU(), nn.Softplus(), CosLayer()]
        elif hasattr(torch.nn, activations):
            acts = [getattr(torch.nn, activations)() for _ in range(9)]
        else:
            raise ValueError('activations must be `basic`, `random`, '\
                             'or else a valid torch.nn activation')
        self.acts = nn.ModuleList(acts)

        if clip_loss:
            self.act_out = torch.tanh
        else:
            self.act_out = torch.sigmoid

    def generate_new_acts(self):
        acts = [randact(activation_set='large') for _ in range(9)]
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
    
    def forward(self, x, y, z, r, noise):
        noise_pt = self.acts[0](self.linear_noise(noise))
        x_pt = self.acts[1](self.linear_x(x))
        y_pt = self.acts[2](self.linear_y(y))
        r_pt = self.acts[3](self.linear_r(r))
        z_pt = self.acts[4](self.linear_z(z))
        r_pt = self.acts[5](self.linear_r(r))
        z = z_pt + x_pt + y_pt + r_pt + noise_pt
        z = self.acts[4](z)
        z = self.acts[5](self.linear1(z))
        z = self.acts[6](self.linear2(z))
        z = self.acts[7](self.linear3(z))
        x = .5 * self.acts[8](self.linear4(z)) + .5
        x = self.act_out(x)
        return x



class INRConvMap3D(nn.Module):
    def __init__(self,
                 noise_dim,
                 c_dim,
                 layer_width,
                 activations='basic',
                 clip_loss=False,
                 name='INRConvMap3D'):
        super(INRConvMap3D, self).__init__()
        self.noise_dim = noise_dim
        self.c_dim = c_dim
        self.layer_width = layer_width
        self.name = name
        self.feat_dim = 24
        self.conv1 = nn.Conv3d(2, self.feat_dim, kernel_size=1, stride=1, padding='same')
        self.conv2 = nn.Conv3d(self.feat_dim, self.feat_dim, kernel_size=1, stride=1, padding='same')
        self.conv3 = nn.Conv3d(self.feat_dim, self.feat_dim, kernel_size=1, stride=1, padding='same')
        self.conv4 = nn.Conv3d(self.feat_dim, self.feat_dim, kernel_size=1, stride=1, padding='same')
        self.conv5 = nn.Conv3d(self.feat_dim, self.feat_dim, kernel_size=1, stride=1, padding='same')
        self.conv6 = nn.Conv3d(self.feat_dim, self.feat_dim, kernel_size=1, stride=1, padding='same')
        self.conv7 = nn.Conv3d(self.feat_dim, self.feat_dim, kernel_size=1, stride=1, padding='same')
        self.conv8 = nn.Conv3d(self.feat_dim, self.feat_dim, kernel_size=1, stride=1, padding='same')
        self.conv_rgb = nn.Conv3d(self.feat_dim, 3, kernel_size=1, stride=1, padding='same')

        if self.activations == 'random':
            acts = [randact(activation_set='large') for _ in range(9)]
        elif self.activations == 'basic':
            acts = [nn.Tanh(), nn.ELU(), nn.Softplus(), nn.Tanh(), SinLayer(),
                    nn.Tanh(), nn.ELU(), nn.Softplus(), CosLayer()]
        elif hasattr(torch.nn, activations):
            acts = [getattr(torch.nn, activations)() for _ in range(9)]
        else:
            raise ValueError('activations must be `basic`, `random`, '\
                             'or else a valid torch.nn activation')
        self.acts = nn.ModuleList(acts)

        self.norms = nn.ModuleList([nn.Identity() for _ in range(8)])

        if clip_loss:
            self.act_out = torch.tanh
        else:
            self.act_out = torch.sigmoid

    def generate_new_acts(self):
        acts = [randact(activation_set='large') for _ in range(9)]
        self.acts = nn.ModuleList(acts)    
    
    def forward(self, x, y, r, noise, extra=None):
        raise NotImplementedError
        x = torch.stack([x, y, r, noise], 0).unsqueeze(0)
        #x = torch.stack([x, y], 0).unsqueeze(0)
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
    times = []
    for _ in range(20):
        model = INRRandomGraph3D()
    print(model)