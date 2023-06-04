import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx
import collections

from neural_canvas.models.ops import (
    ScaleAct,
    AddAct,
    Gaussian,
    CosLayer,
    SinLayer
)


Node = collections.namedtuple('Node', ['id', 'inputs', 'type'])


def get_graph_info(graph):
    input_nodes = []
    output_nodes = []
    nodes = []
    #print ('nodes', graph.number_of_nodes())
    for node in range(graph.number_of_nodes()):
        tmp = list(graph.neighbors(node))
        tmp.sort()
        type = -1
        if node < tmp[0]:
            input_nodes.append(node)
            type = 0
        if node > tmp[-1]:
            output_nodes.append(node)
            type = 1
        nodes.append(Node(node, [n for n in tmp if n < node], type))
    return nodes, input_nodes, output_nodes


def build_random_graph(nodes, input_nodes, output_nodes, p, k):
    g = networkx.random_graphs.connected_watts_strogatz_graph(
        nodes,
        k, p,
        tries=200)
    assert nodes > input_nodes, "number of nodes must be > input ndoes"
    unodes = np.random.randint(input_nodes, nodes, size=(nodes+input_nodes,))
    lnodes = np.random.randint(0, input_nodes, size=(nodes+input_nodes,))
    # make sure input nodes don't have edges between them
    for i in range(input_nodes):
        for j in range(input_nodes):
            try:
                g.remove_edge(i, j)
            except: # no edge exists (easier to ask forgiveness)
                pass
            try:
                g.remove_edge(j, i)
            except:  # no edge exists
                pass
        g.add_edge(i, unodes[i])
    # handle unconnected nodes other than specified input nodes
    # loop through nodes, and add connections from previous nodes if none exist
    for iter, unode in enumerate(range(input_nodes, nodes)):
        if k < input_nodes + k:
            if not any([g.has_edge(lnode, unode) for lnode in range(unode)]):
                n = lnodes[iter] # get one of the input nodes
                g.add_edge(n, unode)
        else:
            if not any([g.has_edge(lnode, unode) for lnode in range(unode)]):
                n = unodes[i+1] # get one of the preceeding nodes
                g.add_edge(n, unode)
                i += 1
    if output_nodes > 1:
        # handle output layers, we want 3 nodes at the output
        # hueristic: try to just use top info, only connect top layers ot 
        for new_node in range(nodes, nodes+output_nodes):
            g.add_node(new_node)
        for node in range(input_nodes, nodes):
            if not any([g.has_edge(lnode, node) for lnode in range(node, nodes)]):
                #output node
                out_node = np.random.choice(np.arange(nodes, nodes+output_nodes))
                g.add_edge(node, out_node)
                #print ('output', node, 'edge: ', node, out_node)
        for out_node in range(nodes, nodes+output_nodes):
            if g.degree[out_node] == 0:
                g.add_edge(np.random.choice(np.arange(nodes//2, nodes)), out_node)
    return g


def plot_graph(g, path=None, plot=False):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    dot = networkx.nx_pydot.to_pydot(g)
    png_str = dot.create_png(prog='dot')
    # treat the DOT output as an image file
    sio = io.BytesIO()
    sio.write(png_str)
    sio.seek(0)
    img = mpimg.imread(sio)
    plt.imshow(img, aspect='equal')
    if path:
        plt.savefig(path)
    if plot:
        plt.show() 
    plt.close('all')


def randact(activation_set='large'):
    if activation_set == 'large':
        acts = [nn.ELU, nn.Hardtanh, nn.LeakyReLU, nn.LogSigmoid,
                nn.SELU, nn.GELU, nn.CELU, nn.Softshrink, nn.Sigmoid,
                SinLayer, CosLayer, nn.Softplus, nn.Mish, nn.Tanh]
    else:
        acts = [nn.Sigmoid, SinLayer, CosLayer, Gaussian, nn.Softplus, nn.Mish,
                nn.Tanh, ScaleAct, AddAct]

    x = torch.randint(0, len(acts), (1,))
    return acts[x]()


class ScaleOp(nn.Module):
    def __init__(self):
        super(ScaleOp, self).__init__()
        r = torch.ones(1,).uniform_(-1, 1)
        self.r = nn.Parameter(r)

    def forward(self, x):
        return x * self.r


class AddOp(nn.Module):
    def __init__(self):
        super(AddOp, self).__init__()
        r = torch.ones(1,).uniform_(-.5, .5)
        self.r = nn.Parameter(r)

    def forward(self, x):
        return x + self.r
    

class LinearActOp(nn.Module):
    def __init__(self, in_d, out_d, actset):
        super(LinearActOp, self).__init__()
        self.linear = nn.Linear(in_d, out_d)
        self.act = randact(actset)

    def forward(self, x):
        return self.act(self.linear(x))


class ConvActOp(nn.Module):
    def __init__(self, in_d, out_d, actset):
        super(ConvActOp, self).__init__()
        self.conv = nn.Conv2d(in_d, out_d, kernel_size=1, stride=1)
        self.act = randact(actset)

    def forward(self, x):
        x = x.reshape(x.size(0), x.size(1), 1, 1)
        out = self.act(self.conv(x))
        out = out.reshape(out.size(0), out.size(1))
        return out


class RandOp(nn.Module):
    def __init__(self, in_dim, out_dim, actset):
        super(RandOp, self).__init__()
        r_id = torch.randint(0, 4, size=(1,))
        if r_id == 0:
            self.op = ScaleOp()
        elif r_id == 1:
            self.op = AddOp()
        elif r_id == 2:
            self.op = LinearActOp(in_dim, out_dim, actset)
        elif r_id == 3:
            self.op = ConvActOp(in_dim, out_dim, actset)
        else:
            raise ValueError

    def forward(self, x):
        return self.op(x)

            
class RandNodeOP(nn.Module):
    def __init__(self, node, in_dim, out_dim, actset):
        super(RandNodeOP, self).__init__()
        self.is_input_node = Node.type == 0
        self.input_nums = len(node.inputs)
        if self.input_nums > 1:
            self.mean_weight = nn.Parameter(torch.ones(self.input_nums))
            self.sigmoid = nn.Sigmoid()
        self.op = RandOp(in_dim, out_dim, actset)

    #TODO test this
    def forward(self, *input):
        if self.input_nums > 1:
            #out = self.sigmoid(self.mean_weight[0]) * input[0]
            out = input[0]
        for i in range(1, self.input_nums):
            #out = out + self.sigmoid(self.mean_weight[i]) * input[i]
            out = out + input[i]
        else:
            out = input[0]
        out = self.op(out)
        return out


class TorchGraph(nn.Module):
    def __init__(self, graph, in_dim, hidden_dim, out_dim, combine, activation_set):
        super(TorchGraph, self).__init__()
        self.nodes, self.input_nodes, self.output_nodes = get_graph_info(graph)
        self.combine = combine
        self.node_ops = nn.ModuleList()
        for node in self.nodes:
            self.node_ops.append(RandNodeOP(node, in_dim, hidden_dim, activation_set))
        if combine:
            self.linear_out = nn.Linear(hidden_dim, out_dim)
            self.act_out = randact(activation_set)
        else:
            self.linear_out = [nn.Linear(hidden_dim, 1) for _ in range(len(
                self.output_nodes))]
            self.act_out = [randact(activation_set) for _ in range(len(self.output_nodes))]
    
    def forward(self, x):
        out = {}
        for id in self.input_nodes:
            out[id] = self.node_ops[id](x)
        for id, node in enumerate(self.nodes):
            if id not in self.input_nodes:
                out[id] = self.node_ops[id](*[out[_id] for _id in node.inputs])
        if self.combine:
            result = out[self.output_nodes[0]]
            for idx, id in enumerate(self.output_nodes):
                if idx > 0:
                    result = result + out[id]
            result = self.act_out(self.linear_out(result))
            return result
        else:
            outputs = [out[id] for id in self.output_nodes]
            outputs = [self.linear_out[i](out[i]) for i in range
                (len(self.output_nodes))]
            result = torch.cat(outputs, dim=-1)
            result = self.act_out[0](result)
        return result
