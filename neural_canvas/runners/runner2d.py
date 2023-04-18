import os
import tqdm
import glob
import shutil

import numpy as np
import torch
import logging

from neural_canvas.utils import utils
from neural_canvas.models.inrf import INRF2D


class RunnerINRF2D:
    def __init__(self, 
                 model=None,
                 output_dir='',
                 save_verbose=False,
                 skip_blank_generations=True,
                 colormaps=None):
        """
        Initialize a runner for INRF2D models
        Provides an interface for generating and regenerating frames, as well as 
        allowing zooming and panning around the scene

        Args:
            model: (INRF2D, optional) model to use for generation
            output_dir: (str, optional) directory to save generated frames to
            save_verbose: (bool, optional) whether to save the frames with verbose names
            skip_blank_generations: (bool, optional) whether to skip blank frames
            colormaps: (list, optional) list of colormaps to use for generation

        """
        if model:
            assert isinstance(model, INRF2D)
        self.model = model
        if save_verbose:
            self.save_prefix = f'z-{model.latent_dim}_scale-{model.latent_scale}-{model.seed}'
        else:
            self.save_prefix = 'gen_image'
        self.save_verbose = save_verbose
        self.skip_blank_generations = skip_blank_generations
        self.colormaps = colormaps
        self.output_dir = output_dir
        logging.basicConfig(level=logging.NOTSET)
        self.logger = logging.getLogger('Runner2D')
        self.logger.setLevel(logging.INFO)

    def backup_pyfiles(self):
        """Copy all python files to the output directory for reference"""
        py_files = glob.glob('*.py')
        assert len(py_files) > 0
        for fn in py_files:
            shutil.copy(fn, os.path.join(self.output_dir, fn))
    
    @torch.no_grad()
    def run_frames(self,
                   latents=None,
                   num_samples=1,
                   zoom_schedule=None,
                   pan_schedule=None,
                   splits=1, 
                   autosave=True):
        """
        Generate frames from a trained model
        Args:
            latents: (torch.tensor, optional) latent vector to use for generation
            num_samples: (int, optional) number of frames to generate
            zoom_schedule: (list, optional) list of zoom values to use for generation
            pan_schedule: (list, optional) list of pan values to use for generation
            splits: (int, optional) number of splits to use for generation
            autosave: (bool, optional) whether to save the frames to disk after generation
        Returns:
            frames: (list) list of generated frames
        """
        # Hueristics based on 32GB of system memory
        #if self.model.x_dim > 256 and self.model.x_dim <= 512:
        #    splits = 16 - (self.model.x_dim % 16)
        #elif self.model.x_dim <= 1024:
        #    splits = 128 - (self.model.x_dim % 512)
        #else:
        #    splits = 1
        assert self.model, 'Must provide a model to generate from'
        frames = []
        metadata = []
        randID = torch.randint(0, 99999999, size=(1,)).item()
        self.logger.info(f'Generating {num_samples} frames using {splits} splits')
        for i in tqdm.tqdm(range(num_samples)):
            if zoom_schedule is not None:
                zoom = zoom_schedule[i]
            else:
                zoom = (.5, .5)
            if pan_schedule is not None:
                pan = pan_schedule[i]
            else:
                pan = (2, 2)
            latents_gen, inputs, meta_latents = self.model.init_latent_inputs(latents=latents,
                                                                              zoom=zoom,
                                                                              pan=pan) 
            frame = self.model.generate(latents_gen, inputs, splits)
            frame = frame.reshape(-1, self.model.x_dim, self.model.y_dim, self.model.c_dim)

            if frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame[0]
            frame = frame.cpu().numpy()
            if self.model.map_fn.final_activation == 'sigmoid':
                frame = (frame * 255.).astype(np.uint8)
            elif self.model.map_fn.final_activation == 'tanh':
                frame = ((frame + 1.) * 127.5).astype(np.uint8)
            else:
                frame = (frame * 255).astype(np.uint8)
            if self.skip_blank_generations:
                if np.abs(frame.max() - frame.min()) < 15:
                    self.logger.info('skipping blank output')
                    continue
            metadata.append(self.model._metadata(meta_latents))
            if autosave:
                save_fn = os.path.join(self.output_dir, f'{self.save_prefix}_{randID}_{i}')
                if self.save_verbose:
                    self.logger.info(f'saving TIFF/PNG images at: {save_fn} of size: {frame.shape}')
                utils.write_image(path=save_fn, img=frame, suffix='png', colormaps=self.colormaps)
                utils.write_image(path=save_fn, img=frame, suffix='tif', metadata=metadata[-1])
            frames.append(frame)
        return frames, metadata
    
    def reinit_model_from_metadata(self, path, output_shape):
        """
        Reinitialize a model from a metadata file

        Args:
            path: (str) path to metadata file
            output_shape: (tuple[int]) new shape (3,) for the model
        Returns:
            latent: (torch.FloatTensor) latent vector used for generation    
        """
        _, metadata = utils.load_tif_metadata(path)
        assert len(output_shape) == 3, f'Invalid output shape: {output_shape}'
        model = INRF2D(output_shape=output_shape,
                       output_dir=self.output_dir,
                       latent_dim=metadata['latent_dim'],
                       latent_scale=metadata['latent_scale'],
                       seed=metadata['seed'])
        model.init_map_fn(mlp_layer_width=metadata['mlp_layer_width'],
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
        return metadata['latent']

    def regen_frames(self,
                     path,
                     output_shape,
                     num_samples=1,
                     splits=1,
                     zoom_schedule=None,
                     pan_schedule=None,
                     save_video=False):
        """
        Regenerate frames from a trained model using a different output shape, 
        Zooming and panning around the scene is also possible by providing a zoom/pan schedule

        Args:
            paths: (str) path to image or directory of images
            output_shape: (tuple[int]) new shape for the model
            num_samples: (int, optional) number of frames to generate
            splits: (int, optional) number of splits to use for generation
            zoom_bounds: (list[float], optional) list of zoom bounds to use for generation
            zoom_scheduler: (str, optional) zoom scheduler to use for generation
            pan_bounds: (list[float], optional) list of pan bounds to use for generation
            pan_scheduler: (str, optional) pan scheduler to use for generation
            save_video: (bool, optional) whether to save the frames to disk as video after generation

        Returns:
            frames: (list) list of generated frames
        """

        assert os.path.exists(path), 'Must provide existing path to image, or directory'
        if os.path.isdir(path):
            image_paths = glob.glob(os.path.join(path, '*.tif'))
        elif os.path.isfile(path):
            image_paths = [path]
        else:
            ValueError(f'Invalid path: {path}')
        for path in image_paths:
            basename = os.path.basename(path)
            save_fn = os.path.join(self.output_dir, f'{basename[:-4]}_reproduce')
            # load metadata from given tif file(s)
            latent = self.reinit_model_from_metadata(path, output_shape)
            frames, metadata = self.run_frames(latent,
                                               num_samples=num_samples,
                                               zoom_schedule=zoom_schedule,
                                               pan_schedule=pan_schedule,
                                               splits=splits,
                                               autosave=False)
            self.logger.info(f'saving {len(frames)} TIFF/PNG images at: {save_fn} of size: {frames[0].shape}')
            for i, (frame, meta) in enumerate(zip(frames, metadata)):
                utils.write_image(path=f'{save_fn}_{i}', img=frame, suffix='png', colormaps=self.colormaps)
                utils.write_image(path=f'{save_fn}_{i}', img=frame, suffix='tif', metadata=meta)
            if save_video:
                utils.write_video(frames, self.output_dir, save_name=basename)
            frames = None
        return frames