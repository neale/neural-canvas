import os
import tqdm
import glob
import shutil

import numpy as np
import torch
import logging

from neural_canvas.utils import utils
from lightning_models import LightningModel3D


class RunnerLightning3D:
    def __init__(self, 
                 model=None,
                 output_dir='',
                 save_verbose=False,
                 skip_blank_generations=True,
                 colormaps=None):
        """
        Initialize a runner for INRF2D models
        Provides an interface for generating and regenerating volumes, as well as 
        allowing zooming and panning around the scene

        Args:
            model: (INRF3D) model to use for generation
            output_dir: (str) directory to save generated volumes tocd ../
            save_verbose: (bool, optional) whether to save the volumes with verbose names
            skip_blank_generations: (bool, optional) whether to skip blank volumes
            colormaps: (list, optional) list of colormaps to use for generation

        """
        if model:
            assert isinstance(model, LightningModel3D)
        self.model = model
        if save_verbose:
            self.save_prefix = f'z-{model.latent_dim}_scale-{model.latent_scale}-{model.seed}'
        else:
            self.save_prefix = 'gen_volume'
        self.save_verbose = save_verbose
        self.skip_blank_generations = skip_blank_generations
        self.colormaps = colormaps
        self.output_dir = output_dir
        logging.basicConfig(level=logging.NOTSET)
        self.logger = logging.getLogger('LightningRunner3D')
        self.logger.setLevel(logging.INFO)


    def backup_pyfiles(self):
        """Copy all python files to the output directory for reference"""
        py_files = glob.glob('*.py')
        assert len(py_files) > 0
        for fn in py_files:
            shutil.copy(fn, os.path.join(self.model.output_dir, fn))
    
    @torch.no_grad()
    def run_volumes(self,
                    latents=None,
                    num_samples=1,
                    zoom_schedule=None,
                    pan_schedule=None,
                    splits=1, 
                    autosave=True):
        """
        Generate volumes from a trained model
        Args:
            latents: (dicts, optional) latent vector information to use for generation
            num_samples: (int, optional) number of volumes to generate
            zoom_schedule: (list, optional) list of zoom values to use for generation
            pan_schedule: (list, optional) list of pan values to use for generation
            splits: (int, optional) number of splits to use for generation
            autosave: (bool, optional) whether to save the volumes to disk after generation
        Returns:
            volumes: (list) list of generated volumes
        """
        # Hueristics based on 32GB of system memory
        #if self.model.x_dim > 256 and self.model.x_dim <= 512:
        #    splits = 16 - (self.model.x_dim % 16)
        #elif self.model.x_dim <= 1024:
        #    splits = 128 - (self.model.x_dim % 512)
        #else:
        #    splits = 1
        assert self.model, 'Must provide a model to generate from, self.model is None'
        volumes = []
        metadata = []
        self.logger.info(f'Generating {num_samples} volumes using {splits} splits')
        for i in tqdm.tqdm(range(num_samples)):
            if zoom_schedule is not None:
                zoom = zoom_schedule[i]
            else:
                zoom = (.5, .5, .5)
            if pan_schedule is not None:
                pan = pan_schedule[i]
            else:
                pan = (2, 2, 2)
            output_shape = (self.model.x_dim, self.model.y_dim, self.model.z_dim)
            latents = self.model.sample_latents(reuse_latents=latents, output_shape=output_shape)
            fields = self.model.construct_fields(output_shape=output_shape, zoom=zoom, pan=pan) 

            volume = self.model.generate(latents, fields, splits=splits)          
            if volume.ndim == 5 and volume.shape[0] == 1:
                volume = volume[0]
            if self.skip_blank_generations:
                if np.abs(volume.max() - volume.min()) < 20:
                    self.logger.info('skipping blank output')
                    continue
            metadata.append(self.model._metadata(latents))
            volumes.append(volume)

        return volumes, metadata
    
    def reinit_model_from_metadata(self, output_shape, path=None, metadata=None):
        """
        Reinitialize a model from a metadata file

        Args:
            path: (str) path to metadata file
            output_shape: (tuple[int]) new shape (4,) for the model
        Returns:
            latents: (dict) latent vector information used for generation    
        """
        if metadata is None:
            _, metadata = utils.load_tif_metadata(path)
        assert len(output_shape) == 4, f'Invalid output shape: `{output_shape}`, need ndim=4'
        model = LightningModel3D(output_shape=output_shape,
                       output_dir=self.output_dir,
                       latent_dim=metadata['latent_dim'],
                       latent_scale=metadata['latent_scale'],
                       seed=metadata['seed'])
        model.init_map_fn(mlp_layer_width=metadata['mlp_layer_width'],
                          conv_feature_map_size=metadata['conv_feature_map_size'],
                          input_encoding_dim=metadata['input_encoding_dim'],
                          activations=metadata['activations'],
                          final_activation=metadata['final_activation'],
                          weight_init=metadata['weight_init'],
                          weight_init_mean=metadata['weight_init_mean'],
                          weight_init_std=metadata['weight_init_std'],
                          weight_init_max=metadata['weight_init_max'],
                          weight_init_min=metadata['weight_init_min'],
                          graph_topology=metadata['graph_topology'],
                          num_graph_nodes=metadata['num_graph_nodes'],
                          graph=metadata['graph'],)
        self.model = model
        loaded_latents = {'sample': metadata['latents'], 
                          'base_shape': (metadata['x_dim'], 
                                         metadata['y_dim'], 
                                         metadata['z_dim']),
                          'sample_shape': (metadata['x_dim'],
                                           metadata['y_dim'],
                                           metadata['z_dim'])}
        return loaded_latents

    def regen_volume(self,
                      path,
                      output_shape,
                      num_samples=1,
                      splits=1,
                      zoom_schedule=None,
                      pan_schedule=None,
                      save_video=False):
        """
        Regenerate single volume from a trained model using a different output shape, 
        Zooming and panning around the scene is also possible by providing a zoom/pan schedule

        Args:
            paths: (str) path to image/volume tiff file
            output_shape: (tuple[int]) new shape for the model
            num_samples: (int, optional) number of volumes to generate
            splits: (int, optional) number of splits to use for generation
            zoom_bounds: (list[float], optional) list of zoom bounds to use for generation
            zoom_scheduler: (str, optional) zoom scheduler to use for generation
            pan_bounds: (list[float], optional) list of pan bounds to use for generation
            pan_scheduler: (str, optional) pan scheduler to use for generation
            save_video: (bool, optional) whether to save the volumes to disk as video after generation

        Returns:
            volumes: (list) list of generated volumes
        """

        assert os.path.exists(path), 'Must provide existing path to volume'
        assert os.path.isfile(path), f'Invalid path: {path}'
        basename = os.path.basename(path)
        save_fn = os.path.join(self.output_dir, f'{basename[:-4]}_reproduce')
        if os.path.exists(save_fn+'_top_view_0.jpg'):
            self.logger.info(f'Found existing output at: {save_fn}')
            return 
        else:
            print ('Running reproduction for: ', path, 'saving at: ', save_fn)
        # load metadata from given tif file(s)
        latents = self.reinit_model_from_metadata(path=path, output_shape=output_shape)
        latents = self.model.sample_latents(reuse_latents=latents)
        volumes, metadata, _ = self.run_volumes(latents=latents,
                                                num_samples=num_samples,
                                                zoom_schedule=zoom_schedule,
                                                pan_schedule=pan_schedule,
                                                splits=splits,
                                                autosave=False)
        self.logger.info(f'saving {len(volumes)} TIFF/PNG images at: {save_fn}')
        for i, (volume, meta) in enumerate(zip(volumes, metadata)):
            if self.save_verbose:
                self.logger.info(f'saving TIFF/PNG images at: {save_fn} of size: {volume.shape}')
            utils.write_image(path=save_fn+f'_front_view_{i}', img=volume[0, :, :, :], suffix='jpg')
            utils.write_image(path=save_fn+f'_side_view_{i}',  img=volume[:, 0, :, :], suffix='jpg')
            utils.write_image(path=save_fn+f'_top_view_{i}',   img=volume[:, :, 0, :], suffix='jpg')
            utils.write_image(path=save_fn, img=volume, suffix='tif', metadata=meta)
        if save_video:
            utils.write_video(volumes, self.output_dir, save_name=basename)
        return volumes
   