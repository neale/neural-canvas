import os
import tqdm
import glob
import shutil

import torch
import logging
import warnings
import numpy as np

from neural_canvas.utils import utils
from neural_canvas.utils.positional_encodings import FourierEncoding
from neural_canvas.losses import losses
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
            latents: (dict, optional) latent vector information to use for generation
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
            output_shape = (self.model.x_dim, self.model.y_dim)
            latents = self.model.sample_latents(reuse_latents=latents, output_shape=output_shape)
            fields = self.model.construct_fields(output_shape=output_shape, zoom=zoom, pan=pan) 
            frame = self.model.generate(latents, fields, splits=splits)
            if frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame[0]     
            if self.skip_blank_generations:
                if np.abs(frame.max() - frame.min()) < 20:
                    self.logger.info('skipping blank output')
                    continue
            metadata.append(self.model._metadata(latents))
            if autosave:
                save_fn = os.path.join(self.output_dir, f'{self.save_prefix}_{randID}_{i}')
                if self.save_verbose:
                    self.logger.info(f'saving TIFF/PNG images at: {save_fn} of size: {frame.shape}')
                utils.write_image(path=save_fn, img=frame, suffix='png', colormaps=self.colormaps)
                utils.write_image(path=save_fn, img=frame, suffix='tif', metadata=metadata[-1])
            frames.append(frame)
        return frames, metadata
    
    def reinit_model_from_metadata(self, output_shape, path=None, metadata=None):
        """
        Reinitialize a model from a metadata file

        Args:
            path: (str) path to metadata file
            output_shape: (tuple[int]) new shape (3,) for the model
        Returns:
            latents: (dict) latent vector information used for generation    
        """
        if metadata is None:
            _, metadata = utils.load_tif_metadata(path)
        assert len(output_shape) == 3, f'Invalid output shape: {output_shape}, need ndim=3'
        model = INRF2D(output_shape=output_shape,
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
                          'sample_shape': (metadata['x_dim'], metadata['y_dim'])}
        return loaded_latents

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
            if basename.endswith('reproduce'):
                basename = basename.split('_reproduce')[0]
            save_fn = os.path.join(self.output_dir, f'{basename[:-4]}_reproduce')
            if os.path.exists(save_fn+'_0_rgb.png'):
                self.logger.info(f'Found existing output at: {save_fn}')
                continue
            else:
                print ('Running reproduction for: ', path, 'saving at: ', save_fn)
            # load metadata from given tif file(s)
            latents = self.reinit_model_from_metadata(path=path, output_shape=output_shape)
            latents = self.model.sample_latents(reuse_latents=latents)
            frames, metadata = self.run_frames(latents=latents,
                                               num_samples=num_samples,
                                               zoom_schedule=zoom_schedule,
                                               pan_schedule=pan_schedule,
                                               splits=splits,
                                               autosave=False)
            if len(frames) == 0:
                self.logger.info(f'No/Bad frames generated for: {path}')
                continue
            self.logger.info(f'saving {len(frames)} TIFF/PNG images at: {save_fn} of size: {frames[0].shape}')
            for i, (frame, meta) in enumerate(zip(frames, metadata)):
                utils.write_image(path=f'{save_fn}_{i}', img=frame, suffix='png', colormaps=self.colormaps)
                utils.write_image(path=f'{save_fn}_{i}', img=frame, suffix='tif', metadata=meta)
            if save_video:
                utils.write_video(frames, self.output_dir, save_name=basename)
            frames = None
        return frames
    
    def fit(self,
            target,
            output_shape,
            loss_weights,
            num_epochs=50,
            num_iters_per_epoch=100,
            lr=1e-3,
            weight_decay=1e-5,
            write_outputs=False):
        """fits model to target image
        Args:
            target: (torch.Tensor) target image
            output_shape: (tuple[int]) new shape for the model
            loss_weights: (dict) dictionary of loss weights
            num_epochs: (int, optional) number of epochs to train for
            n_iters_per_epoch: (int, optional) number of iterations per epoch
            lr: (float, optional) learning rate
            weight_decay: (float, optional) weight decay
            write_outputs: (bool, optional) whether to write outputs to disk
        """
        warnings.warn('Fitting on CPU, will be slow')
        assert self.model is not None, 'Must initialize model before fitting'
        assert target.ndim == 4, 'Target must be 4D tensor'
        optimizer = torch.optim.AdamW(self.model.map_fn.parameters(), 
                                      lr=lr,
                                      weight_decay=weight_decay,
                                      betas=(.9, .999),
                                      eps=1e-7) # support future half precision
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=200, T_mult=2)
        loss = losses.LossModule(**loss_weights)
        epoch_iterator = tqdm.tqdm(range(num_epochs))
        latents = self.model.sample_latents(output_shape=target.shape[2:])
        fields = self.model.construct_fields(output_shape=target.shape[2:])
        
        test_latents = self.model.sample_latents(output_shape=output_shape[:-1])
        test_fields = self.model.construct_fields(output_shape=output_shape[:-1])

        loss_vals = []
        for epoch in epoch_iterator:
            frame, test_frame, loss_val = self.model.fit(target[0],
                                                         num_iters_per_epoch,
                                                         loss,
                                                         optimizer=optimizer,
                                                         scheduler=scheduler,
                                                         inputs=(latents, fields),
                                                         test_inputs=(test_latents, test_fields),
                                                         test_resolution=output_shape,)
            epoch_iterator.set_description(f'Epoch: {epoch}, Loss: {loss_val:.4f}')
            loss_vals.append(loss_val.item())

            if write_outputs:
                if frame.ndim == 4 and frame.shape[0] == 1:
                    frame = frame[0]
                if test_frame.ndim == 4 and test_frame.shape[0] == 1:
                    test_frame = test_frame[0]
                metadata = self.model._metadata(latent=latents)
                test_metadata = self.model._metadata(latent=test_latents)
                utils.write_image(path=f'{self.output_dir}/fit_{epoch}', img=frame, 
                    suffix='png')
                utils.write_image(path=f'{self.output_dir}/fit_{epoch}', img=frame, 
                    suffix='tif', metadata=metadata)
                utils.write_image(path=f'{self.output_dir}/fit_{epoch}_test', img=test_frame, 
                    suffix='png')
                utils.write_image(path=f'{self.output_dir}/fit_{epoch}_test', img=test_frame, 
                    suffix='tif', metadata=test_metadata)
        self.logger.info(f'Finished fitting model')
        return loss_vals

            
