import torch
import torch.nn as nn
import networkx

from neural_canvas.models.ops import *
from neural_canvas.models.torchgraph import randact, build_random_graph, TorchGraph


class INRRandomGraph(nn.Module):
    def __init__(self, 
                 latent_dim,
                 c_dim,
                 layer_width,
                 num_graph_nodes,
                 graph=None,
                 activations='basic',
                 final_activation='sigmoid',
                 name='INRRandomGraph'):
        super(INRRandomGraph, self).__init__()
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.layer_width = layer_width
        self.input_nodes = 1
        self.name = name
        self.nodes = num_graph_nodes
        self.activations = activations
        self.final_activation = final_activation

        self.name = name
        self.linear_latents = nn.Linear(self.latent_dim, self.layer_width)
        self.linear_x = nn.Linear(1, self.layer_width, bias=False)
        self.linear_y = nn.Linear(1, self.layer_width, bias=False)
        self.linear_r = nn.Linear(1, self.layer_width, bias=False)
        self.linear1 = nn.Linear(self.layer_width, self.layer_width)
        self.linear2 = nn.Linear(self.layer_width, self.layer_width)
        self.scale = ScaleAct()
        if self.activations == 'random':
            acts = [randact(activation_set='large') for _ in range(6)]
        elif self.activations == 'basic':
            acts = [nn.Tanh(), nn.ELU(), nn.Softplus(), nn.Tanh(), 
                    Gaussian(), SinLayer()]
        elif hasattr(torch.nn, activations):
            acts = [getattr(torch.nn, activations)() for _ in range(6)]
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
        if final_activation == 'tanh':
            self.act_out = torch.tanh
        elif final_activation == 'sigmoid':
            self.act_out = torch.sigmoid
        elif final_activation is None:
            self.act_out = nn.Identity()
        
    def generate_act_list(self):
        acts = [randact(activation_set='large') for _ in range(6)]
        self.acts = nn.ModuleList(acts)    

    def get_graph_str(self):
        s = ''.join(networkx.generate_graphml(self.graph))
        return s

    def load_graph_str(self, s):
        return networkx.parse_graphml(s, node_type=int)

    def forward(self, inputs, latents):
        x, y, r = inputs[:, 0, ...], inputs[:, 1, ...], inputs[:, 2, ...]
        latents_ = self.acts_start[0](self.linear_latents(latents))
        r_ = self.acts_start[1](self.linear_r(r))
        y_ = self.acts_start[2](self.linear_y(y))
        x_ = self.acts_start[3](self.linear_x(x))
        f = self.acts_start[4](x_+ y_+ r_ + latents_)
        f = self.acts_start[5](self.linear1(f))
        res = self.scale(self.network(f))
        res = self.act_out(res)
        return res
   

class INRLinearMap(nn.Module):
    def __init__(self,
                 latent_dim,
                 c_dim,
                 layer_width,
                 activations='basic',
                 final_activation='sigmoid',
                 name='INRLinearMap'):
        super(INRLinearMap, self).__init__()
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.layer_width = layer_width
        self.activations = activations
        self.final_activation = final_activation

        self.name = name

        self.linear_latents = nn.Linear(self.latent_dim, self.layer_width)
        self.linear_x = nn.Linear(1, self.layer_width, bias=False)
        self.linear_y = nn.Linear(1, self.layer_width, bias=False)
        self.linear_r = nn.Linear(1, self.layer_width, bias=False)

        self.linear1 = nn.Linear(self.layer_width, self.layer_width)
        self.linear2 = nn.Linear(self.layer_width, self.layer_width)
        self.linear3 = nn.Linear(self.layer_width, self.layer_width)
        self.linear4 = nn.Linear(self.layer_width, self.c_dim)

        if self.activations == 'random':
            acts = [randact(activation_set='large') for _ in range(5)]
        elif self.activations == 'basic':
            acts = [nn.GELU(), nn.Softplus(), nn.Tanh(), SinLayer(), ScaleAct()]
        elif hasattr(torch.nn, activations):
            acts = [getattr(torch.nn, activations)() for _ in range(5)]
        else:
            raise ValueError('activations must be `basic`, `random`, '\
                             'or else a valid torch.nn activation')
        self.acts = nn.ModuleList(acts)

        if final_activation == 'tanh':
            self.act_out = torch.tanh
        elif final_activation == 'sigmoid':
            self.act_out = torch.sigmoid
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
 
    def forward(self, inputs, latents):
        x, y, r = inputs[:, 0, ...], inputs[:, 1, ...], inputs[:, 2, ...]
        latents_pt = self.linear_latents(latents)
        x_pt = self.linear_x(x)
        y_pt = self.linear_y(y)
        r_pt = self.linear_r(r)
        z = latents_pt + x_pt + y_pt + r_pt
        z = self.acts[0](z)
        z = self.acts[1](self.linear1(z))
        z = self.acts[2](self.linear2(z))
        z = self.acts[3](self.linear3(z))
        z = self.acts[4](self.linear4(z))
        z_out = self.act_out(z)
        return z_out


class INRConvMap(nn.Module):
    def __init__(self,
                 latent_dim,
                 c_dim,
                 feature_dim,
                 activations='basic',
                 final_activation='sigmoid',
                 name='INRConvMap'):
        super(INRConvMap, self).__init__()
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.feature_dim = feature_dim
        self.activations = activations
        self.final_activation = final_activation
        self.input_channels = latent_dim + c_dim
        self.name = name

        self.conv1 = nn.Conv2d(self.input_channels, self.feature_dim, kernel_size=1, stride=1, padding='same')
        self.conv2 = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, stride=1, padding='same')
        self.conv3 = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, stride=1, padding='same')
        self.conv4 = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, stride=1, padding='same')
        self.conv5 = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, stride=1, padding='same')
        self.conv6 = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, stride=1, padding='same')
        self.conv7 = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, stride=1, padding='same')
        self.conv8 = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, stride=1, padding='same')
        self.conv_rgb = nn.Conv2d(self.feature_dim, c_dim, kernel_size=1, stride=1, padding='same')

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

        if final_activation == 'tanh':
            self.act_out = torch.tanh
        elif final_activation == 'sigmoid':
            self.act_out = torch.sigmoid
        elif final_activation is None:
            self.act_out = nn.Identity()

    def generate_new_acts(self):
        acts = [randact(activation_set='large') for _ in range(9)]
        self.acts = nn.ModuleList(acts)    
    
    def forward(self, inputs, latents):
        x = torch.cat([inputs, latents], 1)
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
        model = INRRandomGraph()
    print(model)