import os
import glob
import torch
import logging 

from neural_canvas.models.inr_maps_2d import *
from neural_canvas.models.inr_maps_3d import *

from neural_canvas.models.weight_inits import *
from neural_canvas.models.inrf_base import INRFBase
from neural_canvas.utils.positional_encodings import coordinates_2D, coordinates_3D


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
        elif len(output_shape) == 4:
            self.x_dim, self.y_dim, self.z_dim, self.c_dim, = output_shape
        else:
            raise ValueError('output_shape must be of length 2, 3, or 4') 
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

        self.default_fields = None        
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
            metadata['latent'] = latent.detach().cpu().tolist()

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
        data = self.map_fn(fields=self.default_fields, latents=self.init_latent_inputs())
        return data
    
    @property
    def fields(self):
        # returns the fields characterized by this INRF
        return {'x': self.default_fields[0, 0],
                'y': self.default_fields[0, 1],
                'r': self.default_fields[0, 2]}    
    
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
        self.default_latents, self.default_fields, _ = self.init_latent_inputs()


    def init_latent_inputs(self,
                           latents=None,
                           batch_size=1,
                           output_shape=None,
                           zoom=(.5,.5),
                           pan=(2,2)):
        """
        initializes latent inputs for the forward map
        Args:
            z (torch tensor, Optional): latent inputs if initialized previously, if provided then 
                this will reshape the latent inputs to the correct dimensions
            batch_size (int): batch size
            output_shape (tuple(int), Optional): dimensions of forward map output. If not provided,
                then the dimensions of the forward map output are assumed to be
                `self.x_dim` and `self.y_dim`
        Returns:
            latents (torch tensor): latent inputs
            fields (torch.tensor) of size (B, N, H, W) that represents a Batch of N inputs
        """
        if batch_size is None:
            batch_size = self.batch_size
        if output_shape is not None:
            assert len(output_shape) == 2 or len(output_shape) == 3, 'output_shape must be of length 2 or 3' \
                f'got `{output_shape}`'
            x_dim, y_dim = output_shape[:2]
        else:
            x_dim, y_dim = self.x_dim, self.y_dim
        if latents is None:
            if self.graph_topology == 'conv':
                latents = torch.ones(batch_size, self.latent_dim, x_dim, y_dim).uniform_(-2.0, 2.0)
            else:
                latents = torch.ones(batch_size, 1, self.latent_dim).uniform_(-2.0, 2.0)
        else:
            if self.graph_topology == 'conv':
                latents = torch.ones(batch_size, self.latent_dim, x_dim, y_dim).uniform_(-2.0, 2.0)
                #assert latents.shape == (batch_size, self.latent_dim, self.x_dim, self.y_dim), '' \
                #    f'need latent shape `{(batch_size, self.latent_dim, self.x_dim, self.y_dim)}`, ' \
                #    f'got `{latents.shape}`'
            else:
                assert latents.shape == (batch_size, 1, self.latent_dim), f'got latent shape {latents.shape}'
        latents_to_save = latents.clone().detach().cpu()

        latents = latents.to(self.device)
        if self.graph_topology != 'conv': # then reshape to long flat vector
            latents = latents.reshape(batch_size, 1, self.latent_dim)
            one_vec = torch.ones(x_dim*y_dim, 1).float().to(self.device)
            latents = (latents * one_vec * self.latent_scale).unsqueeze(0)
        if self.default_fields is None or self.default_fields.shape[-2:] != (x_dim, y_dim):
            self.logger.debug('Detected missing or incompatible 2D fields, initializing ...')
            fields = coordinates_2D(x_dim, 
                                    y_dim,
                                    batch_size=batch_size,
                                    zoom=zoom,
                                    pan=pan,
                                    scale=self.latent_scale,
                                    as_mat=True)#self.graph_topology=='conv_fixed')
            fields = torch.stack(fields, 0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        else:
            fields = self.default_fields
        return latents.to(self.device), fields.to(self.device), latents_to_save

    def generate(self,
                 latents=None,
                 fields=None,
                 output_shape=None,
                 splits=1):
        """
        samples from the forward map
        Args:
            latents (torch tensor): latent inputs
            fields (torch.tensor) of size (B, N, H, W) that represents a Batch of N inputs
                that may represent X, Y, R, or any other input of shape (H, W) that matches
                the desired output shape. 
            splits (int): number of splits to use for sampling. Used to reduce memory usage
        Returns:
            frame (torch tensor): sampled frame
        """
        if output_shape is not None: # ignore passed in parameters
            assert isinstance(output_shape, (tuple, list)), 'output_shape must be a tuple or list' \
                f' got `{type(output_shape)}`'
            latents, fields, _ = self.init_latent_inputs(output_shape=output_shape)
        if fields is not None:
            assert isinstance(fields, torch.Tensor), 'fields must be a torch tensor' \
                f' got {type(fields)}'
        else:
            fields = self.default_fields
        if latents is not None:
            assert isinstance(latents, torch.Tensor), 'latents must be a torch tensor' \
                f' got {type(latents)}'
        else:
            latents = self.default_latents

        batch_size = fields.shape[0]
        n_pts = np.prod(fields.shape[2:])
        output_shape = fields.shape[-2:] + (self.c_dim,)
        if 'mlp' in self.graph_topology or 'WS' in self.graph_topology:
            latents = latents.reshape(-1, self.latent_dim)
            fields = fields.reshape(batch_size, fields.shape[1], -1)
            fields = fields.transpose(0, 2)
        if splits == 1:
            frame = self.map_fn(fields, latents)
        elif splits > 1:
            latents = torch.split(latents, latents.shape[0]//splits, dim=0)
            fields = torch.split(fields, fields.shape[0]//splits, dim=0)
            for i in range(splits):
                f = self.map_fn(fields[i], latents[i])
                torch.save(f, os.path.join(self.tmp_dir, self.output_dir, f'inrf_temp_gen{i}.pt'))

            frame = torch.load(os.path.join(self.tmp_dir, self.output_dir, f'inrf_temp_gen0.pt'))
            for j in range(1, splits):
                frame = torch.cat(
                    [frame, 
                     torch.load(os.path.join(self.tmp_dir, self.output_dir, f'inrf_temp_gen{j}.pt'))],
                    dim=0)

            temp_files = glob.glob(f'{self.tmp_dir}/{self.output_dir}/inrf_temp_gen*')
            for temp in temp_files:
                os.remove(temp)
        else:
            raise ValueError(f'splits must be >= 1, got {splits}')
        frame = frame.reshape(batch_size, *output_shape)
        self.logger.debug(f'Output Frame Shape: {frame.shape}')
        return frame

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
        #if target.max() > 1:
        #    target = target / 127.5 - 1
        assert target.ndim == 3, f'target must have (H, W, C) dimensions, got {target.shape}'
        target = target.to(self.device)
        self.x_dim, self.y_dim, self.c_dim = target.shape

        if inputs is None:
            latents, fields, mlatents = self.init_latent_inputs(
                output_shape=target.shape[1:])
        else:
            latents, fields = inputs
        if test_inputs is None:
            test_latents, test_fields, _ = self.init_latent_inputs(
                mlatents, output_shape=test_resolution)
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
                 output_shape=(512, 512, 512, 3),
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
            raise ValueError('output_shape must be of length 3 or 4 for 3D INRF') 
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

        self.default_fields = None       
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
            metadata['latent'] = latent.detach().cpu().tolist()
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
        data = self.map_fn(fields=self.default_fields, latents=self.init_latent_inputs())
        return data
    
    @property
    def default_fields(self):
        # returns the fields characterized by this INRF
        return {'x': self.default_fields[0, 0],
                'y': self.default_fields[0, 1],
                'z': self.default_fields[0, 2],
                'r': self.default_fields[0, 3]}
    
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
        if graph_topology.lower() == 'mlp':
            map_fn = INRLinearMap3D(
                self.latent_dim, self.c_dim, layer_width=mlp_layer_width,
                activations=activations, final_activation=final_activation)

        elif graph_topology.lower() == 'conv':
            map_fn = INRConvMap3D(
                self.latent_dim, self.c_dim, self.latent_scale, feature_dim=conv_feature_map_size,
                activations=activations, final_activation=final_activation)
            
        elif graph_topology.lower() == 'WS':
            map_fn = INRRandomGraph3D(
                self.latent_dim, self.c_dim, layer_width=mlp_layer_width,
                num_graph_nodes=num_graph_nodes, graph=graph,
                activations=activations, final_activation=final_activation)
        else:
            raise NotImplementedError(f'Graph topology `{graph_topology}` not implemented')

        # initialize weights
        if weight_init == 'normal':
            map_fn = init_weights_normal(map_fn, weight_init_mean, weight_init_std)
        elif weight_init == 'uniform':
            map_fn = init_weights_uniform(map_fn, weight_init_min, weight_init_max)
        elif weight_init == 'dip':
            map_fn = init_weights_dip(map_fn)
        elif weight_init == 'siren':
            map_fn = init_weights_siren(map_fn)
        else:
            raise NotImplementedError(f'weight init {weight_init} not implemented')
        self.mlp_layer_width = mlp_layer_width
        self.conv_feature_map_size = conv_feature_map_size
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
        self.default_latents, self.default_fields, _ = self.init_latent_inputs()

    def init_latent_inputs(self,
                           latents=None,
                           batch_size=1,
                           output_shape=None,
                           zoom=(.5,.5,.5),
                           pan=(2,2,2)):
        """
        initializes latent inputs for the forward map
        Args:
            z (torch tensor, Optional): latent inputs if initialized previously, if provided then 
                this will reshape the latent inputs to the correct dimensions
            batch_size (int): batch size
            output_shape (tuple(int), Optional): dimensions of forward map output. If not provided,
                then the dimensions of the forward map output are assumed to be
                `self.x_dim` and `self.y_dim`
        Returns:
            latents (torch tensor): latent inputs
            fields (torch.tensor) of size (B, N, H, W) that represents a Batch of N inputs
        """
        if batch_size is None:
            batch_size = self.batch_size
        if output_shape is not None:
            assert len(output_shape) == 3 or len(output_shape) == 4, 'output_shape must be of length 3 or 4'
            x_dim, y_dim, z_dim = output_shape[:3]
        else:
            x_dim, y_dim, z_dim = self.x_dim, self.y_dim, self.z_dim
        if latents is None:
            latents = torch.ones(batch_size, 1, self.latent_dim).uniform_(-2., 2.)
            latents_to_save = latents.clone().detach().cpu()
        else:
            assert fields.shape == (batch_size, 1, self.latent_dim)
        latents_to_save = latents.clone().detach().cpu()

        latents = latents.to(self.device)
        latents = latents.reshape(batch_size, 1, self.latent_dim)
        one_vec = torch.ones(x_dim*y_dim*z_dim, 1).float().to(self.device)
        latents = (latents * one_vec * self.latent_scale).unsqueeze(0)
        if self.default_fields is None or self.default_fields.shape[-3:] != (x_dim, y_dim, z_dim):
            self.logger.info('Detected missing or incompatible 3D fields, initializing ...')
            fields = coordinates_3D(x_dim, 
                                    y_dim,
                                    z_dim,
                                    batch_size=batch_size,
                                    zoom=zoom,
                                    pan=pan,
                                    scale=self.latent_scale,
                                    as_mat=False)
            fields = torch.stack(fields, 0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        else:
            fields = self.default_fields
        return latents.to(self.device), fields.to(self.device), latents_to_save

    def generate(self,
                 latents,
                 fields,
                 splits=1):
        """
        samples from the forward map
        Args:
            latents (torch tensor): latent inputs
            fields (torch.tensor) of size (B, N, H, W) that represents a Batch of N inputs
                that may represent X, Y, Z, R, or any other input of shape (H, W, D) that matches
                the desired output shape. 
            splits (int): number of splits to use for sampling. Used to reduce memory usage
        Returns:
            frame (torch tensor): sampled frame
        """
        if fields is not None:
            assert isinstance(fields, torch.Tensor), 'fields must be a torch tensor' \
                f' got `{type(fields)}`'
        else:
            fields = self.default_fields
        if latents is not None:
            assert isinstance(latents, torch.Tensor), 'latents must be a torch tensor' \
                f' got `{type(latents)}`'
        else:
            latents = self.default_latents

        batch_size = fields.shape[0]
        n_pts = np.prod(fields.shape[2:])
        if 'mlp' in self.graph_topology or 'WS' in self.graph_topology:
            latents = latents.reshape(-1, self.latent_dim)
            fields = fields.reshape(-1, *fields.shape[2:])

        if splits == 1:
            volume = self.map_fn(fields, latents)
        elif splits > 1:
            one_vec = torch.ones(n_pts, 1, dtype=torch.float).to(self.device)
            n_pts_split = n_pts // splits
            one_vecs = torch.split(one_vec, len(one_vec)//splits, dim=0)
            fields = torch.split(fields, fields.shape[1]//splits, dim=1)
            for i, one_vec in enumerate(one_vecs):
                latents_reshape = latents.view(batch_size, 1, self.latents_dim) 
                latents_one_vec_i = latents_reshape * one_vec * self.latents_scale
                latents_scale_i = latents_one_vec_i.view(batch_size*n_pts_split, self.latents_dim)
                # forward split through map_fn
                f = self.map_fn(fields[i], latents_scale_i)
                torch.save(f, os.path.join(self.tmp_dir, self.output_dir, f'inrf_temp_gen{i}.pt'))
            volume = torch.load(os.path.join(self.tmp_dir, self.output_dir, f'inrf_temp_gen0.pt'))
            for j in range(1, splits):
                volume = torch.cat(
                    [volume, 
                     torch.load(os.path.join(self.tmp_dir, self.output_dir, f'inrf_temp_gen{j}.pt'))],
                    dim=0)

            temp_files = glob.glob(f'{self.tmp_dir}/{self.output_dir}/inrf_temp_gen*')
            for temp in temp_files:
                os.remove(temp)
        else:
            raise ValueError(f'splits must be >= 1, got {splits}')
        self.logger.debug(f'Output Frame Shape: {volume.shape}')
        return volume
