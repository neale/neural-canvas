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
                 latent_dim=8.0,
                 latent_scale=1.0,
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
        self.map_fn = None
        self.inputs = None

        self.logger = logging.getLogger('INRF2D')
        self.logger.setLevel(logging.INFO)
        self._init_random_seed(seed=seed)
        self._init_paths()

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
    
    def init_map_fn(self,
                    mlp_layer_width=32,
                    conv_feature_map_size=64,
                    activations='fixed',
                    final_activation='sigmoid',
                    weight_init='normal',
                    graph_topology='fixed',
                    num_graph_nodes=10,
                    weight_init_mean=0,
                    weight_init_std=1,
                    weight_init_min=-2,
                    weight_init_max=2,
                    graph=None):
        """
        initializes the forward map function of the implicit neural representation
        Args: 
            activation_set (str): 'fixed', 'random', or instance of torch.nn: denotes the type
                of activation functions used in the forward map
            graph_topology (str): 'mlp_fixed', 'conv_fixed', or 'WS': denotes the type of graph topology
                used in the forward map
            graph (networkx graph): if graph_topology is 'WS', then this is the graph used in the forward map
                used to reproduce the results of a previous or known graph
        """
        if graph_topology == 'mlp_fixed':
            map_fn = INRLinearMap(
                self.latent_dim, self.c_dim, layer_width=mlp_layer_width,
                activations=activations, final_activation=final_activation)

        elif graph_topology == 'conv_fixed':
            map_fn = INRConvMap(
                self.latent_dim, self.c_dim, self.latent_scale, feature_dim=conv_feature_map_size,
                activations=activations, final_activation=final_activation)
            
        elif graph_topology == 'WS':
            map_fn = INRRandomGraph(
                self.latent_dim, self.c_dim, layer_width=mlp_layer_width,
                num_graph_nodes=num_graph_nodes, graph=graph,
                activations=activations, final_activation=final_activation)
        else:
            raise NotImplementedError(f'graph topology {graph_topology} not implemented')

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

    def set_inputs(self, inputs):
        """sets `inputs` as a class variable for use in `generate` method.
           saves time over repeated calls to `generate`
           
           Args:
               inputs (torch tensor): inputs to forward map
        """
        self.inputs = inputs

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
            inputs (torch.tensor) of size (B, N, H, W) that represents a Batch of N inputs
        """
        if batch_size is None:
            batch_size = self.batch_size
        if output_shape is not None:
            assert len(output_shape) == 2 or len(output_shape) == 3, 'output_shape must be of length 2 or 3'
            x_dim, y_dim = output_shape[:2]
        else:
            x_dim, y_dim = self.x_dim, self.y_dim
        if latents is None:
            latents = torch.ones(batch_size, 1, self.latent_dim).uniform_(-2.0, 2.0)
        else:
            assert latents.shape == (batch_size, 1, self.latent_dim)
        latents_to_save = latents.clone().detach().cpu()

        latents = latents.to(self.device)
        latents = latents.reshape(batch_size, 1, self.latent_dim)
        one_vec = torch.ones(x_dim*y_dim, 1).float().to(self.device)
        latents = (latents * one_vec * self.latent_scale).unsqueeze(0)
        if self.inputs is None or self.inputs.shape[-2:] != (x_dim, y_dim):
            self.logger.info('Detected missing or incompatible 2D inputs, initializing ...')
            inputs = coordinates_2D(x_dim, 
                                    y_dim,
                                    batch_size=batch_size,
                                    zoom=zoom,
                                    pan=pan,
                                    scale=self.latent_scale,
                                    as_mat=False)
            inputs = torch.stack(inputs, 0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
        return latents.to(self.device), inputs.to(self.device), latents_to_save

    def generate(self,
                 latents,
                 inputs,
                 splits=1):
        """
        samples from the forward map
        Args:
            latents (torch tensor): latent inputs
            inputs (torch.tensor) of size (B, N, H, W) that represents a Batch of N inputs
                that may represent X, Y, R, or any other input of shape (H, W) that matches
                the desired output shape. 
            splits (int): number of splits to use for sampling. Used to reduce memory usage
        Returns:
            frame (torch tensor): sampled frame
        """
        assert isinstance(inputs, torch.Tensor), 'inputs must be a torch tensor' \
            f' got {type(inputs)}'
        assert isinstance(latents, torch.Tensor), 'latents must be a torch tensor' \
            f' got {type(latents)}'

        batch_size = inputs.shape[0]
        n_pts = np.prod(inputs.shape[2:])
        if 'mlp' in self.graph_topology or 'WS' in self.graph_topology:
            latents = latents.reshape(-1, self.latent_dim)
            inputs = inputs.transpose(1, 2)
            inputs = inputs.reshape(-1, *inputs.shape[2:])

        if splits == 1:
            latents = latents.reshape(batch_size*n_pts, self.latent_dim)
            frame = self.map_fn(inputs, latents)
        elif splits > 1:
            latents = torch.split(latents, latents.shape[0]//splits, dim=0)
            inputs = torch.split(inputs, inputs.shape[0]//splits, dim=0)
            for i in range(splits):
                f = self.map_fn(inputs[i], latents[i])
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
        self.logger.debug(f'Output Frame Shape: {frame.shape}')
        return frame


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
                 latent_dim=8.0,
                 latent_scale=1.0,
                 output_shape=(512, 512, 512, 3),
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
        self.map_fn = None
        self.inputs = None

        self.logger = logging.getLogger('INRF3D')
        self.logger.setLevel(logging.INFO)
        self._init_random_seed(seed=seed)
        self._init_paths()

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
    
    def init_map_fn(self,
                    mlp_layer_width=32,
                    conv_feature_map_size=64,
                    activations='fixed',
                    final_activation='sigmoid',
                    weight_init='normal',
                    graph_topology='fixed',
                    num_graph_nodes=10,
                    weight_init_mean=0,
                    weight_init_std=1,
                    weight_init_min=-2,
                    weight_init_max=2,
                    graph=None):
        """
        initializes the forward map function of the implicit neural representation
        Args: 
            activation_set (str): 'fixed', 'random', or instance of torch.nn: denotes the type
                of activation functions used in the forward map
            graph_topology (str): 'mlp_fixed', 'conv_fixed', or 'WS': denotes the type of graph topology
                used in the forward map
            graph (networkx graph): if graph_topology is 'WS', then this is the graph used in the forward map
                used to reproduce the results of a previous or known graph
        """
        if graph_topology == 'mlp_fixed':
            map_fn = INRLinearMap3D(
                self.latent_dim, self.c_dim, layer_width=mlp_layer_width,
                activations=activations, final_activation=final_activation)

        elif graph_topology == 'conv_fixed':
            map_fn = INRConvMap3D(
                self.latent_dim, self.c_dim, self.latent_scale, feature_dim=conv_feature_map_size,
                activations=activations, final_activation=final_activation)
            
        elif graph_topology == 'WS':
            map_fn = INRRandomGraph3D(
                self.latent_dim, self.c_dim, layer_width=mlp_layer_width,
                num_graph_nodes=num_graph_nodes, graph=graph,
                activations=activations, final_activation=final_activation)
        else:
            raise NotImplementedError(f'graph topology {graph_topology} not implemented')

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

    def set_inputs(self, inputs):
        """sets `inputs` as a class variable for use in `generate` method.
           saves time over repeated calls to `generate`
           
           Args:
               inputs (torch tensor): inputs to forward map
        """
        self.inputs = inputs

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
            inputs (torch.tensor) of size (B, N, H, W) that represents a Batch of N inputs
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
            assert inputs.shape == (batch_size, 1, self.latent_dim)
        latents_to_save = latents.clone().detach().cpu()

        latents = latents.to(self.device)
        latents = latents.reshape(batch_size, 1, self.latent_dim)
        one_vec = torch.ones(x_dim*y_dim*z_dim, 1).float().to(self.device)
        latents = (latents * one_vec * self.latent_scale).unsqueeze(0)
        if self.inputs is None or self.inputs.shape[-3:] != (x_dim, y_dim, z_dim):
            self.logger.info('Detected missing or incompatible 3D inputs, initializing ...')
            inputs = coordinates_3D(x_dim, 
                                    y_dim,
                                    z_dim,
                                    batch_size=batch_size,
                                    zoom=zoom,
                                    pan=pan,
                                    scale=self.latent_scale,
                                    as_mat=False)
            inputs = torch.stack(inputs, 0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        return latents.to(self.device), inputs.to(self.device), latents_to_save

    def generate(self,
                 latents,
                 inputs,
                 splits=1):
        """
        samples from the forward map
        Args:
            latents (torch tensor): latent inputs
            inputs (torch.tensor) of size (B, N, H, W) that represents a Batch of N inputs
                that may represent X, Y, Z, R, or any other input of shape (H, W, D) that matches
                the desired output shape. 
            splits (int): number of splits to use for sampling. Used to reduce memory usage
        Returns:
            frame (torch tensor): sampled frame
        """
        assert isinstance(inputs, torch.Tensor), 'inputs must be a torch tensor' \
            f' got {type(inputs)}'
        assert isinstance(latents, torch.Tensor), 'latents must be a torch tensor' \
            f' got {type(latents)}'

        batch_size = inputs.shape[0]
        n_pts = np.prod(inputs.shape[2:])
        if 'mlp' in self.graph_topology or 'WS' in self.graph_topology:
            latents = latents.reshape(-1, self.latent_dim)
            inputs = inputs.reshape(-1, *inputs.shape[2:])

        if splits == 1:
            volume = self.map_fn(inputs, latents)
        elif splits > 1:
            one_vec = torch.ones(n_pts, 1, dtype=torch.float).to(self.device)
            n_pts_split = n_pts // splits
            one_vecs = torch.split(one_vec, len(one_vec)//splits, dim=0)
            inputs = torch.split(inputs, inputs.shape[1]//splits, dim=1)
            for i, one_vec in enumerate(one_vecs):
                latents_reshape = latents.view(batch_size, 1, self.latents_dim) 
                latents_one_vec_i = latents_reshape * one_vec * self.latents_scale
                latents_scale_i = latents_one_vec_i.view(batch_size*n_pts_split, self.latents_dim)
                # forward split through map_fn
                f = self.map_fn(inputs[i], latents_scale_i)
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
