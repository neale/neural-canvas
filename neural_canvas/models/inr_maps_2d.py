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
            acts = [randact(activation_set='large') for _ in range(6)]
        elif self.activations == 'fixed':
            acts = [nn.Tanh(), nn.ELU(), nn.Softplus(), nn.Tanh(), 
                    Gaussian(), SinLayer()]
        elif hasattr(torch.nn, activations):
            acts = [getattr(torch.nn, activations)() for _ in range(6)]
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
        
    def generate_act_list(self):
        acts = [randact(activation_set='large') for _ in range(6)]
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
        f = self.acts_start[4](x_+ y_+ r_)
        if latents is not None:
            latents_ = self.acts_start[0](self.linear_latents(latents))
            f = f + latents_
        f = self.acts_start[5](self.linear1(f))
        res = self.scale(self.network(f))
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
            acts = [randact(activation_set='large') for _ in range(5)]
        elif self.activations == 'fixed':
            acts = [nn.GELU(), nn.Softplus(), nn.Tanh(), SinLayer(), ScaleAct()]
        elif hasattr(torch.nn, activations):
            acts = [getattr(torch.nn, activations)() for _ in range(5)]
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
 
    def forward(self, fields, latents=None):
        #TODO refactor this to look better, its clunky to support positional encodings
        # field inputs should probably be a dict, since they have physical relevance
        if fields.ndim == 4: # after positional encoding
            chunk_size = fields.shape[1]//3
            x = fields[:, :chunk_size, :, 0].permute(0, 2, 1)
            y = fields[:, chunk_size:2*chunk_size, :, 0].permute(0, 2, 1)
            r = fields[:, chunk_size*2:, :, 0].permute(0, 2, 1)
        else:
            x, y, r = fields[:, 0, ...], fields[:, 1, ...], fields[:, 2, ...]
        x_pt = self.linear_x(x)
        y_pt = self.linear_y(y)
        r_pt = self.linear_r(r)
        z = x_pt + y_pt + r_pt
        if latents is not None:
            latents_pt = self.linear_latents(latents)
            z = z + latents_pt
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
                 input_encoding_dim,
                 activations='fixed',
                 final_activation='sigmoid',
                 name='INRConvMap'):
        super(INRConvMap, self).__init__()
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.feature_dim = feature_dim
        self.input_encoding_dim = input_encoding_dim
        self.activations = activations
        self.final_activation = final_activation
        self.input_channels = 3*input_encoding_dim + latent_dim
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
        elif self.activations == 'fixed':
            acts = [nn.Tanh(), nn.ELU(), nn.Softplus(), nn.Tanh(), SinLayer(),
                    nn.Tanh(), nn.ELU(), nn.Softplus(), CosLayer()]
        elif hasattr(torch.nn, activations):
            acts = [getattr(torch.nn, activations)() for _ in range(9)]
        else:
            raise ValueError('activations must be `fixed`, `random`, '\
                             f'or else a valid torch.nn activation, got {activations}')
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
    
    def forward(self, fields, latents=None):
        if latents is not None:
            x = torch.cat([fields, latents], 1)
        else:
            x = fields
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
    for _ in range(20):
        model = INRRandomGraph()
    print(model)