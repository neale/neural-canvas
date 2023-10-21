import os
import numpy as np
import torch
import logging 


import neural_canvas.models.weight_inits as weight_inits
from lightning_generators import INRRandomGraph, INRLinearMap
from lightning_model_base import LightningModelBase
from neural_canvas.utils.positional_encodings import coordinates_2D
from neural_canvas.utils import unnormalize_and_numpy


logging.getLogger().setLevel(logging.ERROR)


class LightningModel2D(LightningModelBase):
    """initializes a 2D Implicit Neural Representation (INR) model

    Args:
        latents_dim (int, optional): dimension of latent vector.
        latent_scale (float, optional): scale of latent vector.
        output_shape (tuple[int], optional): shape of output image.
        output_dir (str, optional): directory to save output images.
        tmp_dir (str, optional): directory to save temporary files.
        seed_gen (int, optional): seed for random number generator.
        seed (int, optional): directly set seed.
        device (str, optional): device to run model on.
    """

    def __init__(self,
                 latent_dim=8,
                 latent_scale=1.0,
                 graph_topology='mlp',
                 output_shape=(512, 512, 3),
                 output_dir='./outputs',
                 tmp_dir='./tmp',
                 seed_gen=987654321,
                 seed=None,
                 device='cpu'):
    
        self.latent_dim = latent_dim
        if len(output_shape) == 2:
            self.x_dim, self.y_dim = output_shape
            self.c_dim = 1
        if len(output_shape) == 3:
            self.x_dim, self.y_dim, self.c_dim = output_shape
        else:
            raise ValueError(f'output_shape must be of length 2 or 3, got `{output_shape}`') 
        self.latent_scale = latent_scale
        self.output_dir = output_dir
        self.tmp_dir = tmp_dir
        self.device = device
        self.seed = seed
        self.seed_gen = seed_gen

        self.logger = logging.getLogger('LightningModel2D')
        self.logger.setLevel(logging.INFO)
        self._init_random_seed(seed=seed)
        self._init_paths()

        self.init_map_fn(graph_topology=graph_topology)

    def _init_random_seed(self, seed=None):
        """ 
        initializes random seed for torch. Random seed needs
            to be a stored value so that we can save the right metadata. 
            This is not to be confused with the uid that is not a seed, 
            rather how we associate user sessions with data
        """
        if seed == None:
            self.logger.debug(f'initing with seed gen: {self.seed_gen}')
            self.seed = np.random.randint(self.seed_gen)
            torch.manual_seed(self.seed)
        else:
            torch.manual_seed(self.seed)

    def _init_paths(self):
        """
        initializes paths for saving data
        """
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.tmp_dir, self.output_dir), exist_ok=True)

    def _metadata(self, latent=None):
        """
        returns metadata for the model
        """
        metadata = {
            'latent_dim': self.latent_dim,
            'latent_scale': self.latent_scale,
            'x_dim': self.x_dim,
            'y_dim': self.y_dim,
            'c_dim': self.c_dim,
            'seed': self.seed,
            'device': self.device,
        }
        if self.map_fn is not None:
            metadata['mlp_layer_width'] = self.mlp_layer_width
            metadata['conv_feature_map_size'] = self.conv_feature_map_size
            metadata['input_encoding_dim'] = self.input_encoding_dim
            metadata['activations'] = self.activations
            metadata['final_activation'] = self.final_activation
            metadata['weight_init'] = self.weight_init
            metadata['graph_topology'] = self.graph_topology
            metadata['num_graph_nodes'] = self.num_graph_nodes
            metadata['weight_init_mean'] = self.weight_init_mean
            metadata['weight_init_std'] = self.weight_init_std
            metadata['weight_init_min'] = self.weight_init_min
            metadata['weight_init_max'] = self.weight_init_max
            if hasattr(self.map_fn, 'get_graph_str'):
                metadata['graph'] = self.map_fn.get_graph_str()
        if latent is not None:
            metadata['latents'] = latent['sample'].detach().cpu().tolist()
        return metadata
    
    @property
    def size(self):
        # print number of parameters in map_fn
        if self.map_fn is None:
            num_params = 0
        else:
            num_params = sum(p.numel() for p in self.map_fn.parameters() if p.requires_grad)
        return num_params
    
    @property
    def data(self):
        # returns the data characterized by this INRF
        fields = self.construct_fields()
        data = self.map_fn(fields=fields['coords'])
        return data
    
    @property
    def fields(self):
        # returns the fields characterized by this INRF
        return self.construct_fields() 
    
    def init_map_weights(self):
        # initialize weights
        if self.weight_init == 'normal':
            self.map_fn = weight_inits.init_weights_normal(
                self.map_fn, self.weight_init_mean, self.weight_init_std)
        elif self.weight_init == 'uniform':
            self.map_fn = weight_inits.init_weights_uniform(
                self.map_fn, self.weight_init_min, self.weight_init_max)
        elif self.weight_init == 'xavier':
            self.map_fn = weight_inits.init_weights_xavier(
                self.map_fn, self.weight_init_min, self.weight_init_max)
        elif self.weight_init == 'kaiming':
            self.map_fn = weight_inits.init_weights_kaiming(
                self.map_fn, self.weight_init_min, self.weight_init_max)        
        else:
            self.logger.info(f'weight init `{self.weight_init}` not implemented')
        torch.nn.init.orthogonal_(self.map_fn.linear_out.weight.data, 
                                    gain=torch.nn.init.calculate_gain('tanh')) 
        self.map_fn = self.map_fn.to(self.device)

    def init_map_fn(self,
                    mlp_layer_width=32,
                    conv_feature_map_size=32,
                    input_encoding_dim=1,
                    activations='fixed',
                    final_activation='sigmoid',
                    layer_aggregation='linear',
                    weight_init='normal',
                    graph_topology='mlp',
                    num_graph_nodes=10,
                    weight_init_mean=0,
                    weight_init_std=1,
                    weight_init_min=-2,
                    weight_init_max=2,
                    graph=None):
        """
        initializes the forward map function of the implicit neural representation
        Args: 
            mlp_layer_width (int): width of the layers in the MLP
            conv_feature_map_size (int): number of feature maps in the convolutional layers
            input_encoding_dim (int): dimension of the input encoding
            activations (str): 'fixed', 'random', or instance of torch.nn: denotes the type
                of activation functions used in the forward map
            final_activation (str): 'sigmoid', 'tanh', 'relu', or instance of torch.nn: denotes
                the type of activation function used in the final layer of the forward map
            weight_init (str): 'normal', 'uniform', or instance of torch.nn.init: denotes the
                type of weight initialization used in the forward map
            graph_topology (str): 'mlp', 'conv', 'ws': denotes the type
                of graph topology used in the forward map
            num_graph_nodes (int): number of nodes in the graph
            weight_init_mean (float): mean of the weight initialization
            weight_init_std (float): standard deviation of the weight initialization
            weight_init_min (float): minimum value of the weight initialization
            weight_init_max (float): maximum value of the weight initialization
            num_fourier_freqs (int, Optional): number of fourier frequencies to use in
                the input encoding
            num_siren_layers (int, Optional): number of siren layers to use in the input encoding
            siren_scale (float, Optional): scale of the siren layers
            siren_scale_init (float, Optional): scale of the siren layers at the first layer
            graph (torch.Tensor): networkx string representation of the graph
        """
        if graph_topology == 'mlp':
            map_fn = INRLinearMap(
                self.latent_dim, self.c_dim, layer_width=mlp_layer_width,
                input_encoding_dim=input_encoding_dim,
                activations=activations, final_activation=final_activation,
                layer_aggregation=layer_aggregation)

            
        elif graph_topology == 'WS':
            map_fn = INRRandomGraph(
                self.latent_dim, self.c_dim, layer_width=mlp_layer_width,
                input_encoding_dim=input_encoding_dim,
                num_graph_nodes=num_graph_nodes, graph=graph,
                activations=activations, final_activation=final_activation)

        else:
            raise NotImplementedError(f'Graph topology `{graph_topology}` not implemented')
    
        self.fourier_encoding = None

        # initialize weights
        self.mlp_layer_width = mlp_layer_width
        self.conv_feature_map_size = conv_feature_map_size
        self.input_encoding_dim = input_encoding_dim
        self.activations = activations
        self.final_activation = final_activation
        self.layer_aggregation = layer_aggregation
        self.weight_init = weight_init
        self.graph_topology = graph_topology
        self.num_graph_nodes = num_graph_nodes
        self.weight_init_mean = weight_init_mean
        self.weight_init_std = weight_init_std
        self.weight_init_min = weight_init_min
        self.weight_init_max = weight_init_max
        self.num_siren_layers = None
        self.siren_scale = None
        self.siren_scale_init = None
        self.map_fn = map_fn
        self.init_map_weights()

    def construct_fields(self,
                         output_shape=None,
                         zoom=(.5, .5),
                         pan=(2, 2),
                         coord_fn=None):
        if output_shape is not None:
            assert len(output_shape) in [2, 3], 'output_shape must be 2D or 3D'
            if len(output_shape) == 3:
                batch_size = output_shape[0]
            else:
                batch_size = 1
            x_dim, y_dim = output_shape[-2], output_shape[-1]
        else:
            batch_size = 1
            x_dim, y_dim = self.x_dim, self.y_dim
        assert len(zoom) == 2, f'zoom direction must be 2D, got `{zoom}`'
        assert len(pan) == 2, f'pan direction must be 2D, got `{pan}`'
        if coord_fn is not None:
            coords = coord_fn(x_dim, y_dim, batch_size=batch_size)
        else:
            coords = coordinates_2D(x_dim, 
                                    y_dim,
                                    batch_size=batch_size,
                                    zoom=zoom,
                                    pan=pan,
                                    scale=self.latent_scale,
                                    as_mat=True)
            coords = torch.stack(coords, 0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        fields = {'coords': coords.to(self.device),
                  'shape': (batch_size, x_dim, y_dim),
                  'zoom': zoom, 
                  'pan': pan}
        return fields
    
    @torch.no_grad()
    def sample_latents(self, reuse_latents=None, output_shape=None, generator=None):
        """
        initializes latent inputs for the forward map
        Args:
            reuse_latents (dict, Optional): dictionary of latent inputs to reuse
            output_shape (tuple(int), Optional): dimensions of forward map output. If not provided,
                then the dimensions of the forward map output are assumed to be
                `self.x_dim` and `self.y_dim` with an optional batch dimension
        Returns:
            latents (dict): latent input information
        """
        if output_shape is not None:
            assert len(output_shape) in [2, 3], 'output_shape must be 2D or 3D'
            if len(output_shape) == 3:
                batch_size = output_shape[0]
            else:
                batch_size = 1
            x_dim, y_dim = output_shape[-2], output_shape[-1]
        else:
            batch_size = 1
            x_dim, y_dim = self.x_dim, self.y_dim

        if reuse_latents is not None:
            latents = reuse_latents.copy()
        else:
            latents = {'base_shape':(x_dim, y_dim), 'sample_shape': (x_dim, y_dim)}
        if generator is None:
            generator = lambda x: x.uniform_(-2, 2)

        if 'sample' not in latents.keys():
            sample = torch.ones(batch_size, 1, self.latent_dim)
            sample = generator(sample)
            latents['sample'] = sample.to(self.device)
        if self.graph_topology == 'siren':
            latents['input'] = latents['sample']
        else:
            one_vec = torch.ones(x_dim*y_dim, 1).float().to(self.device)
            latents['input'] = torch.matmul(one_vec, latents['sample']) * self.latent_scale
        latents['input'] = latents['input'].to(self.device)
        return latents

    def generate(self, fields, latents):
        """
        samples from the forward map
        Args:
            latents (dict, Optional): latent input information for the forward map
            fields (torch.tensor) of size (B, N, H, W) that represents a Batch of N inputs
                that may represent X, Y, R, or any other input of shape (H, W) that matches
                the desired output shape. 
            output_shape (tuple(int), Optional): dimensions of forward map output. If not provided,
                then the dimensions of the forward map output are assumed to be
                (`self.x_dim`, `self.y_dim`, self.z_dim`)
            splits (int): number of splits to use for sampling. Used to reduce memory usage
            sample_latent (bool): whether to sample random latent inputs
            latent_generator (Callable, Optional): generator to use for sampling latents

        Returns:
            frame (torch tensor): sampled frame
        """
        output_shape = fields['shape'] + (self.c_dim,)

        if latents is not None:
            latent = latents['input']
        latent = latent.reshape(-1, self.latent_dim)

        field = fields['coords']  # [B, NF, H, W]
        field = field.reshape(1, field.shape[1], -1)
        field = field.permute(2, 1, 0)

        # generate
        frame = self.map_fn(field, latent)
        frame = frame.reshape(output_shape)
        return unnormalize_and_numpy(frame)

