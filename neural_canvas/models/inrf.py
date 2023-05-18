import os
import glob
import torch
import logging 

from neural_canvas.models.inr_maps_2d import *
from neural_canvas.models.inr_maps_3d import *

from neural_canvas.models.weight_inits import *
from neural_canvas.models.inrf_base import INRFBase
from neural_canvas.utils.positional_encodings import coordinates_2D, coordinates_3D
from neural_canvas.utils import unnormalize_and_numpy


logging.getLogger().setLevel(logging.ERROR)


class INRF2D(INRFBase):
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

        self.logger = logging.getLogger('INRF2D')
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
        data = self.map_fn(fields['coords'])
        return data
    
    @property
    def fields(self):
        # returns the fields characterized by this INRF
        return self.construct_fields() 
    
    def init_map_weights(self):
        # initialize weights
        if self.weight_init == 'normal':
            self.map_fn = init_weights_normal(
                self.map_fn, self.weight_init_mean, self.weight_init_std)
        elif self.weight_init == 'uniform':
            self.map_fn = init_weights_uniform(
                self.map_fn, self.weight_init_min, self.weight_init_max)
        elif self.weight_init == 'dip':
            self.map_fn = init_weights_dip(self.map_fn)
        elif self.weight_init == 'siren':
            self.map_fn = init_weights_siren(self.map_fn)
        else:
            self.logger.info(f'weight init `{self.weight_init}` not implemented')

    def init_map_fn(self,
                    mlp_layer_width=32,
                    conv_feature_map_size=64,
                    input_encoding_dim=1,
                    activations='fixed',
                    final_activation='sigmoid',
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
            graph (torch.Tensor): networkx string representation of the graph
        """
        if graph_topology == 'simple':
            map_fn = INRLSimpleLinearMap(
                self.latent_dim, self.c_dim, layer_width=mlp_layer_width,
                input_encoding_dim=input_encoding_dim,
                activations=activations, final_activation=final_activation)

        elif graph_topology == 'mlp':
            map_fn = INRLinearMap(
                self.latent_dim, self.c_dim, layer_width=mlp_layer_width,
                input_encoding_dim=input_encoding_dim,
                activations=activations, final_activation=final_activation)

        elif graph_topology == 'conv':
            map_fn = INRConvMap(
                self.latent_dim, self.c_dim, feature_dim=conv_feature_map_size,
                input_encoding_dim=input_encoding_dim,
                activations=activations, final_activation=final_activation)
            
        elif graph_topology == 'WS':
            map_fn = INRRandomGraph(
                self.latent_dim, self.c_dim, layer_width=mlp_layer_width,
                input_encoding_dim=input_encoding_dim,
                num_graph_nodes=num_graph_nodes, graph=graph,
                activations=activations, final_activation=final_activation)
        else:
            raise NotImplementedError(f'Graph topology `{graph_topology}` not implemented')
        # initialize weights
        self.mlp_layer_width = mlp_layer_width
        self.conv_feature_map_size = conv_feature_map_size
        self.input_encoding_dim = input_encoding_dim
        self.activations = activations
        self.final_activation = final_activation
        self.weight_init = weight_init
        self.graph_topology = graph_topology
        self.num_graph_nodes = num_graph_nodes
        self.weight_init_mean = weight_init_mean
        self.weight_init_std = weight_init_std
        self.weight_init_min = weight_init_min
        self.weight_init_max = weight_init_max

        self.map_fn = map_fn.to(self.device)
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
        fields = {'coords': coords,
                  'shape': (batch_size, x_dim, y_dim),
                  'zoom': zoom, 
                  'pan': pan}
        return fields
    
    def sample_latents(self, reuse_latents=None, output_shape=None):
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
            latents = reuse_latents
        else:
            latents = {'base_shape':(x_dim, y_dim), 'sample_shape': (x_dim, y_dim)}
        if self.graph_topology == 'conv':  #
            sample = torch.ones(batch_size, self.latent_dim, x_dim, y_dim)
            latents['sample'] = sample.uniform_(-2, 2)
            latents['input'] = latents['sample'].clone()
        else:
            if 'sample' not in latents.keys():
                sample = torch.ones(batch_size, 1, self.latent_dim)
                sample = sample.uniform_(-2, 2)
                latents['sample'] = sample
            one_vec = torch.ones(x_dim*y_dim, 1).float().to(self.device)
            latents['input'] = torch.matmul(one_vec, latents['sample']) * self.latent_scale
        latents['input'] = latents['input'].to(self.device)
        return latents

    def generate(self,
                 latents=None,
                 fields=None,
                 output_shape=None,
                 splits=1,
                 sample_latent=False,
                 unnormalize_output=True):
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
            unnormalize_output (bool): whether to unnormalize the output
        Returns:
            frame (torch tensor): sampled frame
        """
        if output_shape is None:
            if fields is None:
                fields = self.construct_fields()
            if latents is None:
                if sample_latent:
                    latents = self.sample_latents(output_shape=fields['shape'])                

        else: 
            if fields is None:
                fields = self.construct_fields(output_shape=output_shape)
            if latents is not None:
                if fields['shape'] != latents['sample_shape']:
                    latents = self.sample_latents(
                        output_shape=output_shape, reuse_latents=latents)
            if sample_latent:
                latents = self.sample_latents(output_shape=output_shape)

        output_shape = fields['shape'] + (self.c_dim,)
        if latents is not None:
            latent = latents['input']
            if self.graph_topology in['mlp', 'WS', 'simple']:
                latent = latent.reshape(-1, self.latent_dim)
            if splits > 1:
                latent = torch.split(latent, latent.shape[0]//splits, dim=0)
        else:
            latent = None

        field = fields['coords']  # [B, NF, H, W]
        if self.graph_topology in['mlp', 'WS', 'simple']:
            field = field.reshape(1, field.shape[1], -1)
            field = field.permute(2, 1, 0)
        if splits == 1:
            frame = self.map_fn(field, latent)
        elif splits > 1:
            field = torch.split(field, field.shape[0]//splits, dim=0)
            for i in range(splits):
                if latent is not None:
                    l_i = latent[i]
                else:
                    l_i = None
                f = self.map_fn(field[i], l_i)
                torch.save(f, os.path.join(self.tmp_dir, self.output_dir, f'inrf2D_temp_gen{i}.pt'))

            frame = torch.load(os.path.join(self.tmp_dir, self.output_dir, f'inrf2D_temp_gen0.pt'))
            for j in range(1, splits):
                frame = torch.cat(
                    [frame, 
                     torch.load(os.path.join(self.tmp_dir, self.output_dir, f'inrf2D_temp_gen{j}.pt'))],
                    dim=0)

            temp_files = glob.glob(f'{self.tmp_dir}/{self.output_dir}/inrf2D_temp_gen*')
            for temp in temp_files:
                os.remove(temp)
        else:
            raise ValueError(f'splits must be >= 1, got {splits}')

        if self.graph_topology in ['mlp', 'WS', 'simple']:
            frame = frame.reshape(output_shape)
        elif self.graph_topology == 'conv':
            frame = frame.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        #if unnormalize_output:
        #    frame = unnormalize_and_numpy(frame, self.map_fn.final_activation)
        self.logger.debug(f'Output Frame Shape: {frame.shape}')
        return (frame*255).numpy().astype(np.uint8)

    def fit(self,
            target,
            n_iters=201,
            loss=None,
            optimizer=None,
            scheduler=None,
            inputs=None, 
            test_inputs=None,
            test_resolution=(512, 512, 3)):
        """optimizes parameters of 2D INRF to fit a target image
        Args:
            target (torch tensor or np.ndarray): target image to fit
            n_iters (int): number of iterations to train
            loss (torch.nn loss function): loss function to use
            optimizer (torch.optim optimizer): optimizer to use
            scheduler (torch.optim scheduler): scheduler to use
            inputs (tuple): tuple of (latents, fields) to use as inputs
            test_inputs (tuple): tuple of (latents, fields) to use as inputs for testing
            test_resolution (tuple): resolution of test inputs
        Returns:
            frame (torch tensor): generated frame after fitting
            test_frame (torch tensor): generated frame of different resolution
            loss (float): final loss value
        """
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float()

        assert target.ndim == 3, f'target must have (C, H, W) dimensions, got {target.shape}'
        target = target.to(self.device)
        self.c_dim, self.x_dim, self.y_dim = target.shape

        if inputs is None:
            latents = self.sample_latents(output_shape=target.shape[1:])
            fields = self.construct_fields(output_shape=target.shape[1:])
        else:
            latents, fields = inputs
        if test_inputs is None:
            test_latents = self.sample_latents(output_shape=test_resolution[:-1])   
            test_fields = self.construct_fields(output_shape=test_resolution[:-1])
        else:
            test_latents, test_fields = test_inputs
        assert isinstance(target, (torch.Tensor, np.ndarray)), 'target must' \
            f'be a torch tensor or numpy array got `{type(target)}`'

        if loss is None:
            loss = nn.L1Loss()
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                self.map_fn.parameters(), lr=5e-3, weight_decay=1e-5)

        target = target.unsqueeze(0)
        for it in range(n_iters):
            optimizer.zero_grad()
            frame = self.map_fn(fields['coords'], latents['input'])
            frame = frame.reshape(target.shape)
            loss_val = loss(frame, target)
            loss_val.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
        test_frame = self.map_fn(test_fields['coords'], test_latents['input'])
        test_frame = test_frame.permute(0, 2, 3, 1)
        test_frame = unnormalize_and_numpy(test_frame, self.map_fn.final_activation)
        frame = unnormalize_and_numpy(frame[0].permute(1, 2, 0), self.map_fn.final_activation)
        return frame, test_frame, loss_val


class INRF3D(INRFBase):
    """initializes a 3D Implicit Neural Representation (INR) model
    
    Args:
        latent_dim (int, optional): dimensionality of latent vector
        latent_scale (float, optional): scale of latent vector
        output_shape (tuple[int], optional): shape of output image
        output_dir (str, optional): directory to save outputs
        tmp_dir (str, optional): directory to save temporary files
        seed_gen (int, optional): seed for random number generator
        seed (int, optional): set seed directly
        device (str, optional): device to use for torch
    """
    def __init__(self,
                 latent_dim=8,
                 latent_scale=1.0,
                 output_shape=(256,256,256,3),
                 graph_topology='mlp',
                 output_dir='./outputs',
                 tmp_dir='./tmp',
                 seed_gen=987654321,
                 seed=None,
                 device='cpu'):
    
        self.latent_dim = latent_dim
        if len(output_shape) == 3:
            self.x_dim, self.y_dim, self.z_dim = output_shape
            self.c_dim = 1
        if len(output_shape) == 4:
            self.x_dim, self.y_dim, self.z_dim, self.c_dim = output_shape
        else:
            raise ValueError(f'output_shape must be of length 3 or 4, got `{len(output_shape)}`') 
        self.latent_scale = latent_scale
        self.output_dir = output_dir
        self.tmp_dir = tmp_dir
        self.device = device
        self.seed = seed
        self.seed_gen = seed_gen
        
        self.logger = logging.getLogger('INRF3D')
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
            'z_dim': self.z_dim,
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
            self.map_fn = init_weights_normal(
                self.map_fn, self.weight_init_mean, self.weight_init_std)
        elif self.weight_init == 'uniform':
            self.map_fn = init_weights_uniform(
                self.map_fn, self.weight_init_min, self.weight_init_max)
        elif self.weight_init == 'dip':
            self.map_fn = init_weights_dip(self.map_fn)
        elif self.weight_init == 'siren':
            self.map_fn = init_weights_siren(self.map_fn)
        else:
            self.logger.info(f'weight init `{self.weight_init}` not implemented')

    def init_map_fn(self,
                    mlp_layer_width=32,
                    conv_feature_map_size=64,
                    input_encoding_dim=1,
                    activations='fixed',
                    final_activation='sigmoid',
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
            graph (torch.Tensor): networkx string representation of the graph
        """
        if graph_topology == 'mlp':
            map_fn = INRLinearMap3D(
                self.latent_dim, self.c_dim, layer_width=mlp_layer_width,
                activations=activations, final_activation=final_activation)

        elif graph_topology == 'conv':
            map_fn = INRConvMap3D(
                self.latent_dim, self.c_dim, self.latent_scale, feature_dim=conv_feature_map_size,
                activations=activations, final_activation=final_activation)
            
        elif graph_topology == 'WS':
            map_fn = INRRandomGraph3D(
                self.latent_dim, self.c_dim, layer_width=mlp_layer_width,
                num_graph_nodes=num_graph_nodes, graph=graph,
                activations=activations, final_activation=final_activation)
        else:
            raise NotImplementedError(f'Graph topology `{graph_topology}` not implemented')

        self.mlp_layer_width = mlp_layer_width
        self.conv_feature_map_size = conv_feature_map_size
        self.input_encoding_dim = input_encoding_dim
        self.activations = activations
        self.final_activation = final_activation
        self.weight_init = weight_init
        self.graph_topology = graph_topology
        self.num_graph_nodes = num_graph_nodes
        self.weight_init_mean = weight_init_mean
        self.weight_init_std = weight_init_std
        self.weight_init_min = weight_init_min
        self.weight_init_max = weight_init_max

        self.map_fn = map_fn.to(self.device)
        self.init_map_weights()

    def construct_fields(self,
                         output_shape=None,
                         zoom=(.5, .5, .5),
                         pan=(2, 2, 2),
                         coord_fn=None):
        assert len(output_shape) in [3, 4], f'output_shape must be 2D or 3D, got `{output_shape}`'
        if len(output_shape) == 4:
            batch_size = output_shape[0]
        else:
            batch_size = 1
        x_dim, y_dim, z_dim = output_shape[-3], output_shape[-2], output_shape[-1]
        assert len(zoom) == 3, f'zoom direction must be 3D, got `{zoom}`'
        assert len(pan) == 3, f'pan direction must be 3D, got `{pan}`'
        if coord_fn is not None:
            coords = coord_fn(x_dim, y_dim, z_dim, batch_size=batch_size)
        else:
            fields = coordinates_3D(x_dim, 
                                    y_dim,
                                    z_dim,
                                    batch_size=batch_size,
                                    zoom=zoom,
                                    pan=pan,
                                    scale=self.latent_scale,
                                    as_mat=True)
            coords = torch.stack(fields, 0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        fields = {'coords': coords.to(self.device), 
                  'shape': (batch_size, x_dim, y_dim, z_dim),
                  'zoom': zoom,
                  'pan': pan}
        return fields
    
    def sample_latents(self, reuse_latents=None, output_shape=None):
        """
        initializes latent inputs for the forward map
        Args:
            reuse_latents (dict): latent inputs if initialized previously, if provided then 
                this will use the data of the previous inputs to initialize the new inputs
            output_shape (tuple(int), Optional): dimensions of forward map output. If not provided,
                then the dimensions of the forward map output are assumed to be
                `self.x_dim` and `self.y_dim`
        Returns:
            latents (dict): latent input information
        """
        if output_shape is not None:
            assert len(output_shape) in [3, 4], 'output_shape must be 3D or 4D'
            if len(output_shape) == 4:
                batch_size = output_shape[0]
            else:
                batch_size = 1
            x_dim, y_dim, z_dim = output_shape[-3], output_shape[-2], output_shape[-1]
        else:
            batch_size = 1
            x_dim, y_dim, z_dim = self.x_dim, self.y_dim, self.z_dim

        if reuse_latents is not None:
            latents = reuse_latents
        else:
            latents = {'base_shape':(x_dim, y_dim, z_dim),
                       'sample_shape': (x_dim, y_dim, z_dim)}
        if self.graph_topology == 'conv':  #
            sample = torch.ones(batch_size, self.latent_dim, x_dim, y_dim, z_dim)
            latents['sample'] = sample.uniform_(-1, 1)
            latents['inputs'] = latents['sample'].clone()
        else:
            if 'sample' not in latents.keys():
                sample = torch.ones(batch_size, 1, self.latent_dim)
                sample = sample.uniform_(-1, 1)
                latents['sample'] = sample
            latents['input'] = latents['sample'].clone()
            #one_vec = torch.ones(x_dim*y_dim*z_dim, 1).float().to(self.device)
            #latents['input'] = torch.matmul(one_vec, latents['sample']) * self.latent_scale
        latents['input'] = latents['input'].to(self.device)
        return latents
    
    def generate(self,
                 latents=None,
                 fields=None,
                 output_shape=None,
                 splits=1,
                 sample_latent=False,
                 unnormalize_output=True):
        """
        samples from the forward map
        Args:
            latents (torch tensor): latent input information for the forward map
            fields (torch.tensor) of size (B, N, H, W, D) that represents a Batch of N inputs
                that may represent X, Y, Z, R, or any other input of shape (H, W, D) that matches
                the desired output shape. 
            output_shape (tuple(int), Optional): dimensions of forward map output. If not provided,
                then the dimensions of the forward map output are assumed to be
                (`self.x_dim`, `self.y_dim`, self.z_dim`)
            splits (int): number of splits to use for sampling. Used to reduce memory usage
            sample_latent (bool): whether to sample random latent inputs
            unnormalize_output (bool): whether to unnormalize the output
        Returns:
            volume (torch tensor): sampled volume
        """
        if output_shape is None:
            if fields is None:
                fields = self.construct_fields()
            if latents is None:
                latents = self.sample_latents(output_shape=fields['shape'])

        else: 
            fields = self.construct_fields(output_shape=output_shape)
            if latents is not None:
                if fields['shape'] != latents['sample_shape']:
                    latents = self.sample_latents(
                        output_shape=output_shape, reuse_latents=latents)
            if sample_latent:
                latents = self.sample_latents(output_shape=output_shape)

        batch_size = fields['shape'][0]
        n_pts = np.prod(fields['shape'][1:])
        output_shape = fields['shape'] + (self.c_dim,)

        if latents is not None:
            if splits == 1:
                latent = latents['input']
                size = np.prod(latents['base_shape'])
                one_vec = torch.ones(size, 1).float().to(self.device)
                latents['input'] = torch.matmul(one_vec, latent) * self.latent_scale
            latent = latents['input'].view(-1,  self.latent_dim)
        else:
            latent = None

        fields = fields['coords']
        if splits == 1:
            if self.graph_topology in ['mlp', 'WS']:
                fields = fields.reshape(batch_size, fields.shape[1], -1)
                fields = fields.transpose(0, 2)
                print (fields.shape, latent.shape)
            volume = self.map_fn(fields, latent)
        elif splits > 1:
            assert n_pts % splits == 0, 'number of splits must be divisible by number of points'
            fields = torch.split(fields, fields.shape[2]//splits, dim=2)
            for i, field_i in enumerate(fields):
                field_i = field_i.reshape(batch_size, field_i.shape[1], -1)
                field_i = field_i.permute(2, 1, 0)
                one_vec = torch.ones(n_pts//splits, 1, dtype=torch.float).to(self.device)
                if latent is not None:
                    latent_one_vec_i = latent * one_vec * self.latent_scale
                    latent_scale_i = latent_one_vec_i.view(batch_size*(n_pts//splits), self.latent_dim)
                else:
                    latent_scale_i = None
                # forward split through map_fn
                volume = self.map_fn(field_i, latent_scale_i)
                torch.save(volume, os.path.join(self.tmp_dir, self.output_dir, f'inrf3D_temp_gen{i}.pt'))
            volume = torch.load(os.path.join(self.tmp_dir, self.output_dir, f'inrf3D_temp_gen0.pt'))
            for j in range(1, splits):
                volume = torch.cat(
                    [volume, 
                     torch.load(os.path.join(self.tmp_dir, self.output_dir, f'inrf3D_temp_gen{j}.pt'))],
                    dim=0)

            temp_files = glob.glob(f'{self.tmp_dir}/{self.output_dir}/inrf3D_temp_gen*')
            for temp in temp_files:
                os.remove(temp)
        else:
            raise ValueError(f'splits must be >= 1, got {splits}')
        volume = volume.reshape(output_shape)
        if unnormalize_output:
            volume = unnormalize_and_numpy(volume, self.map_fn.final_activation)
        self.logger.debug(f'Output Frame Shape: {volume.shape}')
        return volume

    def fit(self,
            target,
            n_iters=100,
            loss=None,
            optimizer=None,
            scheduler=None,
            inputs=None, 
            test_inputs=None,
            test_resolution=(512, 512, 3)):
        """optimizes parameters of 2D INRF to fit a target image
        Args:
            target (torch tensor or np.ndarray): target image to fit
            n_iters (int): number of iterations to train
            loss (torch.nn loss function): loss function to use
            optimizer (torch.optim optimizer): optimizer to use
            scheduler (torch.optim scheduler): scheduler to use
            inputs (tuple): tuple of (latents, fields) to use as inputs
            test_inputs (tuple): tuple of (latents, fields) to use as inputs for testing
            test_resolution (tuple): resolution of test inputs
        Returns:
            frame (torch tensor): generated frame after fitting
            test_frame (torch tensor): generated frame of different resolution
            loss (float): final loss value
        """
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float()

        assert target.ndim == 4, f'target must have (H, W, D, C) dimensions, got {target.shape}'
        target = target.to(self.device)
        self.x_dim, self.y_dim, self.z_dim, self.c_dim = target.shape

        if inputs is None:
            latents, fields = self.init_latent_inputs(
                output_shape=target.shape[1:])
        else:
            latents, fields = inputs
        if test_inputs is None:
            test_latents, test_fields = self.init_latent_inputs(
                reuse_latents=latents, output_shape=test_resolution)
        else:
            test_latents, test_fields = test_inputs
        assert isinstance(target, (torch.Tensor, np.ndarray)), 'target must' \
            f'be a torch tensor or numpy array got `{type(target)}`'

        if loss is None:
            loss = nn.L1Loss()
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                self.map_fn.parameters(), lr=5e-3, weight_decay=1e-5)

        for _ in range(n_iters):
            optimizer.zero_grad()
            frame = self.map_fn(fields, latents)
            frame = frame.reshape(target.shape)
            loss_val = loss(frame, target)
            loss_val.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        test_frame = self.map_fn(test_fields, test_latents)
        test_frame = test_frame.reshape(test_resolution)
        test_frame = unnormalize_and_numpy(test_frame, self.map_fn.final_activation)
        frame = unnormalize_and_numpy(frame, self.map_fn.final_activation)
        return frame, test_frame, loss_val