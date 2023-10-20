import torch
import torch.nn as nn
import networkx
from einops import rearrange

from neural_canvas.models.ops import (
    ScaleAct,
    Gaussian,
    SinLayer,
    CosLayer,
)
from neural_canvas.models.torchgraph import randact, build_random_graph, TorchGraph


class INRRandomGraph(nn.Module):
    def __init__(self, 
                 latent_dim,
                 c_dim,
                 layer_width,
                 input_encoding_dim,
                 num_graph_nodes,
                 graph=None,
                 activations='fixed',
                 final_activation='sigmoid',
                 name='INRRandomGraph'):
        super(INRRandomGraph, self).__init__()
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.layer_width = layer_width
        self.input_encoding_dim = input_encoding_dim

        self.input_nodes = 1
        self.name = name
        self.nodes = num_graph_nodes
        self.activations = activations
        self.final_activation = final_activation

        self.name = name
        self.linear_latents = nn.Linear(self.latent_dim, self.layer_width)
        self.linear_x = nn.Linear(self.input_encoding_dim, self.layer_width, bias=False)
        self.linear_y = nn.Linear(self.input_encoding_dim, self.layer_width, bias=False)
        self.linear_r = nn.Linear(self.input_encoding_dim, self.layer_width, bias=False)
        self.linear1 = nn.Linear(self.layer_width, self.layer_width)
        self.linear2 = nn.Linear(self.layer_width, self.layer_width)
        self.scale = ScaleAct()
        if self.activations == 'random':
            acts = [randact(activation_set='large') for _ in range(7)]
        elif self.activations == 'fixed':
            acts = [nn.Tanh(), nn.ELU(), nn.Softplus(), nn.Tanh(), 
                    Gaussian(), SinLayer(), nn.Tanh()]
        elif hasattr(torch.nn, activations):
            acts = [getattr(torch.nn, activations)() for _ in range(7)]
        else:
            raise ValueError('activations must be `fixed`, `random`, '\
                             f'or else a valid torch.nn activation, got {activations}')
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
        print (self.graph)
        
    def generate_new_acts(self):
        acts = [randact(activation_set='large') for _ in range(7)]
        self.acts = nn.ModuleList(acts)    

    def get_graph_str(self):
        s = ''.join(networkx.generate_graphml(self.graph))
        return s

    def load_graph_str(self, s):
        return networkx.parse_graphml(s, node_type=int)

    def forward(self, fields, latents=None):
        x, y, r = fields[:, 0, ...], fields[:, 1, ...], fields[:, 2, ...]
        r_ = self.acts_start[1](self.linear_r(r))
        y_ = self.acts_start[2](self.linear_y(y))
        x_ = self.acts_start[3](self.linear_x(x))
        f = (x_+ y_+ r_)
        if latents is not None:
            latents_ = self.acts_start[0](self.linear_latents(latents))
            f = f + latents_
        f = self.acts_start[4](f)
        f = self.acts_start[5](self.linear1(f))
        res = self.acts_start[6](self.scale(self.network(f)))
        res = self.act_out(res)
        return res
  

class INRLinearMap(nn.Module):
    def __init__(self,
                 latent_dim,
                 c_dim,
                 layer_width,
                 input_encoding_dim,
                 activations='fixed',
                 final_activation='sigmoid',
                 name='INRLinearMap'):
        super(INRLinearMap, self).__init__()
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.layer_width = layer_width
        self.input_encoding_dim = input_encoding_dim
        self.activations = activations
        self.final_activation = final_activation

        self.name = name

        self.linear_latents = nn.Linear(self.latent_dim, self.layer_width)
        self.linear_x = nn.Linear(self.input_encoding_dim, self.layer_width, bias=False)
        self.linear_y = nn.Linear(self.input_encoding_dim, self.layer_width, bias=False)
        self.linear_r = nn.Linear(self.input_encoding_dim, self.layer_width, bias=False)

        self.linear1 = nn.Linear(self.layer_width, self.layer_width)
        self.linear2 = nn.Linear(self.layer_width, self.layer_width)
        self.linear3 = nn.Linear(self.layer_width, self.layer_width)
        self.linear4 = nn.Linear(self.layer_width, self.c_dim)

        if self.activations == 'random':
            acts = [randact(activation_set='large') for _ in range(9)]
        elif self.activations == 'fixed':
            acts = [ScaleAct(), nn.Softplus(), nn.Tanh(), SinLayer(), 
                    nn.GELU(), nn.Softplus(), nn.Tanh(), SinLayer(), ScaleAct()]
        elif hasattr(torch.nn, activations):
            acts = [getattr(torch.nn, activations)() for _ in range(9)]
        else:
            raise ValueError('activations must be `fixed`, `random`, '\
                             f'or else a valid torch.nn activation, got {activations}')
        self.acts = nn.ModuleList(acts)

        if final_activation == 'tanh':
            self.act_out = torch.tanh
        elif final_activation == 'sigmoid':
            self.act_out = torch.sigmoid
        elif final_activation is None:
            self.act_out = nn.Identity()

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
 
    def forward(self, fields, latents=None):
        #TODO refactor this to look better, its clunky to support positional encodings
        if fields.ndim == 4: # after positional encoding
            chunk_size = fields.shape[1]//3
            x = fields[:, :chunk_size, :, 0].permute(0, 2, 1)
            y = fields[:, chunk_size:2*chunk_size, :, 0].permute(0, 2, 1)
            r = fields[:, chunk_size*2:, :, 0].permute(0, 2, 1)
        else:
            x, y, r = fields[:, 0, ...], fields[:, 1, ...], fields[:, 2, ...]
        x_pt = self.acts[0](self.linear_x(x))
        y_pt = self.acts[1](self.linear_y(y))
        r_pt = self.acts[2](self.linear_r(r))
        z = x_pt + y_pt + r_pt
        if latents is not None:
            z += self.acts[3](self.linear_latents(latents))
        z = self.acts[4](z)
        z = self.acts[5](self.linear1(z))
        z = self.acts[6](self.linear2(z))
        z = self.acts[7](self.linear3(z))
        z_out = self.act_out(self.linear4(z))
        return z_out


    

if __name__ == '__main__':
    for _ in range(20):
        model = INRRandomGraph()
    print(model)
