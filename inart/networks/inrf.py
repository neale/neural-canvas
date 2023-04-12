import os
import glob
import torch
import logging 

from networks.inr_maps_2d import *
from networks.weight_inits import *


logging.getLogger().setLevel(logging.ERROR)


class INRF2D(object):
    """initializes a 2D Implicit Neural Representation (INR) model"""
    def __init__(self,
                 noise_dim=4,
                 noise_scale=10,
                 n_samples=6,
                 x_dim=512,
                 y_dim=512,
                 c_dim=3,
                 layer_width=4,
                 batch_size=1,
                 weight_init='normal',
                 graph_nodes=10,
                 seed_gen=987654321,
                 seed=None,
                 clip_loss=False,
                 output_dir='.',
                 tmp_dir='./tmp',
                 device='cpu',
                 graph=None):

        self.noise_dim = noise_dim
        self.n_samples = n_samples
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.c_dim = c_dim 
        self.noise_scale = noise_scale
        self.weight_init = weight_init
        self.layer_width = layer_width
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.tmp_dir = tmp_dir

        self.graph_nodes = graph_nodes
        self.graph = graph
        
        self.map_fn = None
        self.clip_loss = clip_loss
        self.device = device
        self.logger = logging.getLogger('INRF2Dlogger')
        self.logger.setLevel(logging.INFO)

        self.seed = seed
        self.seed_gen = seed_gen
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

    def init_map_fn(self,
                    activations='fixed',
                    graph_topology='fixed',
                    graph=None):
        """
        initializes the forward map function of the implicit neural representation
        Args: 
            activations (str): 'fixed' or 'permute' or 'random', denotes the type of activation functions
                used in the forward map
            graph_topology (str): 'fixed' or 'WS' or 'conv_fixed', denotes the type of graph topology
                used in the forward map
            graph (networkx graph): if graph_topology is 'WS', then this is the graph used in the forward map
                used to reproduce the results of a previous or known graph
        """
        if graph_topology == 'mlp_fixed':
            map_fn = INRLinearMap(
                self.noise_dim, self.c_dim, self.layer_width, self.noise_scale, clip_loss=self.clip_loss,
                activations=activations)

        elif graph_topology == 'conv_fixed':
            map_fn = INRConvMap(
                self.noise_dim, self.c_dim, self.layer_width, self.noise_scale, clip_loss=self.clip_loss,
                activations=activations)
            
        elif graph_topology == 'WS':
            map_fn = INRRandomGraph(
                self.noise_dim, self.c_dim, self.layer_width, self.noise_scale, self.graph_nodes,
                graph, activations=activations)
        else:
            raise NotImplementedError

        # initialize weights
        if self.weight_init == 'normal':
            map_fn = init_weights_normal(map_fn, 0, 1)
        elif self.weight_init == 'uniform':
            map_fn = init_weights_uniform(map_fn, -2, 2)
        elif self.weight_init == 'dip':
            map_fn = init_weights_dip(map_fn)
        elif self.weight_init == 'siren':
            map_fn = init_weights_siren(map_fn)
        else:
            raise NotImplementedError
        self.map_fn = map_fn.to(self.device)

    def init_latent_inputs(self, z=None, batch_size=None, x_dim=None, y_dim=None):
        """
        initializes latent inputs for the forward map
        Args:
            z (torch tensor, Optional): latent inputs if initialized previously, if provided then 
                this will reshape the latent inputs to the correct dimensions
            batch_size (int): batch size
            x_dim (int): x dimension of target data
            y_dim (int): y dimension of target data
        Returns:
            inputs_scaled (torch tensor): latent inputs
        """
        if batch_size is None:
            batch_size = self.batch_size
        if x_dim is None:
            x_dim = self.x_dim
        if y_dim is None:
            y_dim = self.y_dim
        if z is None:
            z = torch.ones(batch_size, 1, self.noise_dim).uniform_(-2., 2.)
        else:
            assert z.shape == (batch_size, 1, self.noise_dim)
        z = z.to(self.device)
        z = z.reshape(batch_size, 1, self.noise_dim)
        one_vec = torch.ones(x_dim*y_dim, 1).float().to(self.device)
        inputs_scaled = z * one_vec * self.noise_scale
        return inputs_scaled.unsqueeze(0).float()

    def sample(self, noise, x, y, r, batch_size=1, splits=1):
        """
        samples from the forward map
        Args:
            noise (torch tensor): latent inputs
            x (torch tensor): x coordinates
            y (torch tensor): y coordinates
            r (torch tensor): radius
            batch_size (int): batch size
            splits (int): number of splits to use for sampling
        Returns:
            frame (torch tensor): sampled frame
        """
        if splits == 1:
            n_pts = np.prod(x.shape)
            noise = noise.reshape(batch_size*n_pts, self.noise_dim)
            frame = self.map_fn(x, y, r, noise, extra=None)
        elif splits > 1:
            n_pts_split = n_pts // splits
            one_vecs = torch.split(one_vec, len(one_vec)//splits, dim=0)
            x = torch.split(x, len(x)//splits, dim=0)
            y = torch.split(y, len(y)//splits, dim=0)
            r = torch.split(r, len(r)//splits, dim=0)
            for i, one_vec in enumerate(one_vecs):
                noise_reshape = noise.view(batch_size, 1, self.noise_dim) 
                noise_one_vec_i = noise_reshape * one_vec * self.noise_scale
                noise_scale_i = noise_one_vec_i.view(batch_size*n_pts_split, self.noise_dim)
                # forward split through map_fn
                f = self.map_fn(x[i], y[i], r[i], noise_scale_i, extra=None)
                torch.save(f, os.path.join(self.tmp_dir, self.output_dir, f'inrf_temp_gen{i}.pt'))
            frames = [torch.load(os.path.join(self.tmp_dir, self.output_dir, f'inrf_temp_gen{j}.pt')) for j in range(splits)]
            frame = torch.cat(frames, dim=0)
            temp_files = glob.glob(f'{self.tmp_dir}/{self.output_dir}/inrf_temp_gen*')
            for temp in temp_files:
                os.remove(temp)
        else:
            raise ValueError

        self.logger.debug(f'Output Frame Shape: {frame.shape}')
        return frame

