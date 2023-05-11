import logging
from abc import ABC, abstractmethod

logging.getLogger().setLevel(logging.ERROR)


class INRFBase(ABC):
    def __init__(self,
                 noise_dim=8.0,
                 noise_scale=1.0,
                 output_shape=(512, 512, 3),
                 output_dir='./outputs',
                 tmp_dir='./tmp',
                 seed_gen=987654321,
                 seed=None,
                 device='cpu'):
    
        self.noise_dim = noise_dim
        if len(output_shape) == 2:
            self.x_dim, self.y_dim = output_shape
            self.c_dim = 1
        if len(output_shape) == 3:
            self.x_dim, self.y_dim, self.c_dim = output_shape
        elif len(output_shape) == 4:
            self.x_dim, self.y_dim, self.z_dim, self.c_dim, = output_shape
        else:
            raise ValueError('output_shape must be of length 2, 3, or 4') 
        self.noise_scale = noise_scale
        self.output_dir = output_dir
        self.tmp_dir = tmp_dir
        self.device = device
        self.seed = seed
        self.seed_gen = seed_gen
        self.map_fn = None

        self.logger = logging.getLogger('INRF2Dlogger')
        self.logger.setLevel(logging.INFO)
        self._init_random_seed(seed=seed)
        self._init_paths()

    @abstractmethod
    def _init_random_seed(self, seed=None):
        pass

    @abstractmethod
    def _init_paths(self):
        pass

    @abstractmethod
    def init_map_fn(self, **kwargs):
        pass

    @abstractmethod
    def init_latent_inputs(self, z=None, batch_size=1, output_shape=None):
        pass

    @abstractmethod
    def generate(self, num=1, latents=None, inputs=None, batch_size=1, splits=1):
        pass

    @abstractmethod
    def _metadata(self, latents=None):
        pass

    def __repr__(self):
        return f"INRF(latent_dim={self.latent_dim}, latent_scale={self.latent_scale}, " \
               f"output_shape=({self.x_dim}, {self.y_dim}, {self.c_dim}), output_dir='{self.output_dir}', " \
               f"tmp_dir='{self.tmp_dir}', seed_gen={self.seed_gen}, seed={self.seed}, device='{self.device}')"