import os
import tqdm
import glob
import shutil

import numpy as np
import torch
import logging
import warnings

from neural_canvas.utils import utils
from neural_canvas.utils.positional_encodings import FourierEncoding
from neural_canvas.losses import losses
from neural_canvas.models.inrf import INRF3D


class RunnerINRF3D:
    def __init__(self, 
                 model,
                 output_dir,
                 save_verbose=False,
                 skip_blank_generations=True,
                 colormaps=None):
        """
        Initialize a runner for INRF2D models
        Provides an interface for generating and regenerating volumes, as well as 
        allowing zooming and panning around the scene

        Args:
            model: (INRF3D) model to use for generation
            output_dir: (str) directory to save generated volumes to
            save_verbose: (bool, optional) whether to save the volumes with verbose names
            skip_blank_generations: (bool, optional) whether to skip blank volumes
            colormaps: (list, optional) list of colormaps to use for generation

        """
        assert isinstance(model, INRF3D)
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
        self.logger = logging.getLogger('Runner3D')
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
        randID = torch.randint(0, 99999999, size=(1,)).item()
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
            latents_gen, inputs = self.model.init_latent_inputs(latents=latents,
                                                                zoom=zoom,
                                                                pan=pan) 
            volume = self.model.generate(latents_gen, inputs, splits)
            volume = volume.reshape(-1, self.model.x_dim, self.model.y_dim, self.model.z_dim, self.model.c_dim)
            if volume.ndim == 5 and volume.shape[0] == 1:
                volume = volume[0]
            volume = volume.cpu().numpy()
            if self.model.map_fn.final_activation == 'sigmoid':
                volume = (volume * 255.).astype(np.uint8)
            elif self.model.map_fn.final_activation == 'tanh':
                volume = ((volume + 1.) * 127.5).astype(np.uint8)
            else:
                volume = (volume * 255).astype(np.uint8)
            if self.skip_blank_generations:
                if np.abs(volume.max() - volume.min()) < 5:
                    self.logger.info('skipping blank output')
                    continue
            metadata.append(self.model._metadata(latents_gen))
            if autosave:
                save_fn = os.path.join(self.output_dir, f'{self.save_prefix}_{randID}_{i}')
                if self.save_verbose:
                    self.logger.info(f'saving TIFF/PNG images at: {save_fn} of size: {volume.shape}')
                utils.write_image(path=save_fn+'_front_view', img=volume[0, :, :, :], suffix='jpg')
                utils.write_image(path=save_fn+'_side_view',  img=volume[:, 0, :, :], suffix='jpg')
                utils.write_image(path=save_fn+'_top_view',   img=volume[:, :, 0, :], suffix='jpg')
                utils.write_image(path=save_fn, img=volume, suffix='tif', metadata=metadata[-1])
            volumes.append(volume)
        return volumes, metadata
    
    def reinit_model_from_metadata(self, path, output_shape):
        """
        Reinitialize a model from a metadata file

        Args:
            path: (str) path to metadata file
            output_shape: (tuple[int]) new shape (4,) for the model
        Returns:
            latents: (dict) latent vector information used for generation    
        """
        metadata = utils.load_tif_metadata(path)
        assert len(output_shape) == 4, f'Invalid output shape: `{output_shape}`, need ndim=4'
        model = INRF3D(output_shape=output_shape,
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
        return metadata['latents']

    def regen_volumes(self,
                      path,
                      output_shape,
                      num_samples=1,
                      splits=1,
                      zoom_schedule=None,
                      pan_schedule=None,
                      save_video=False):
        """
        Regenerate volumes from a trained model using a different output shape, 
        Zooming and panning around the scene is also possible by providing a zoom/pan schedule

        Args:
            paths: (str) path to image or directory of images
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

        assert os.path.exists(path), 'Must provide existing path to volume, or directory'
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
            latents = self.reinit_model_from_metadata(path, output_shape)
            volumes, metadata = self.run_volumes(latents=latents,
                                                 num_samples=num_samples,
                                                 zoom_schedule=zoom_schedule,
                                                 pan_schedule=pan_schedule,
                                                 splits=splits,
                                                 autosave=False)
            self.logger.info(f'saving {len(volumes)} TIFF/PNG images at: {save_fn} of size: {volumes[0].shape}')
            for i, (volume, meta) in enumerate(zip(volumes, metadata)):
                utils.write_image(path=f'{save_fn}_{i}', img=volume, suffix='png', colormaps=self.colormaps)
                utils.write_image(path=f'{save_fn}_{i}', img=volume, suffix='tif', metadata=meta)
            if save_video:
                utils.write_video(volumes, self.output_dir, save_name=basename)
            volumes = None
        return volumes
    
    def fit(self,
            target,
            output_shape,
            loss_weights,
            num_epochs=50,
            num_iters_per_epoch=100,
            lr=1e-3,
            weight_decay=1e-5,
            use_fourier_encoding=False,
            num_freqs=5,
            device='cpu'):
        """fits model to target image
        Args:
            target: (torch.Tensor) target image
            output_shape: (tuple[int]) new shape for the model
            loss_weights: (dict) dictionary of loss weights
            num_epochs: (int, optional) number of epochs to train for
            n_iters_per_epoch: (int, optional) number of iterations per epoch
            lr: (float, optional) learning rate
            weight_decay: (float, optional) weight decay
        """
        warnings.warn('Naive 3D volume fitting, will consume large amount of disk space')
        warnings.warn('Fitting on CPU, will be slow')
        assert self.model is not None, 'Must initialize model before fitting'
        optimizer = torch.optim.AdamW(self.model.map_fn.parameters(), 
                                      lr=lr,
                                      weight_decay=weight_decay,
                                      betas=(.9, .999),
                                      eps=1e-7) # support future half precision
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=200, T_mult=2)
        loss = losses.LossModule(**loss_weights)
        epoch_iterator = tqdm.tqdm(range(num_epochs))
        latents, inputs = self.model.init_latent_inputs()
        test_latents, test_inputs, _ = self.model.init_latent_inputs(
            reuse_latents=latents,
            output_shape=output_shape)
        if use_fourier_encoding:
            input_encoding = FourierEncoding(num_freqs).to(device)
            inputs = input_encoding(inputs)
            test_inputs = input_encoding(test_inputs)

        loss_vals = []
        for epoch in epoch_iterator:
            volume, test_volume, loss_val = self.model.fit(target,
                                                          num_iters_per_epoch,
                                                          loss,
                                                          optimizer=optimizer,
                                                          scheduler=scheduler,
                                                          inputs=(latents, inputs),
                                                          test_inputs=(test_latents, test_inputs),
                                                          test_resolution=output_shape,)
            epoch_iterator.set_description(f'Epoch: {epoch}, Loss: {loss_val:.4f}')
            loss_vals.append(loss_val.item())

            if volume.ndim == 5 and volume.shape[0] == 1:
                volume = volume[0]
            if test_volume.ndim == 5 and test_volume.shape[0] == 1:
                test_volume = test_volume[0]
            volume = volume.permute(1, 2, 0)
            test_volume = test_volume.permute(1, 2, 0)
            volume = volume.detach().cpu().numpy()
            test_volume = test_volume.detach().cpu().numpy()
            if self.model.map_fn.final_activation == 'tanh':
                volume = ((volume + 1.) * 127.5).astype(np.uint8)
                test_volume = ((test_volume + 1.) * 127.5).astype(np.uint8)
            else:
                volume = (volume * 255).astype(np.uint8)
                test_volume = (test_volume * 255).astype(np.uint8)

            metadata = self.model._metadata(latent=latents)
            test_metadata = self.model._metadata(latent=test_latents)
            utils.write_image(path=f'{self.output_dir}/fit_{epoch}', img=volume, 
                suffix='png')
            utils.write_image(path=f'{self.output_dir}/fit_{epoch}', img=volume, 
                suffix='tif', metadata=metadata)
            utils.write_image(path=f'{self.output_dir}/fit_{epoch}_test', img=test_volume, 
                suffix='png')
            utils.write_image(path=f'{self.output_dir}/fit_{epoch}_test', img=test_volume, 
                suffix='tif', metadata=test_metadata)
        self.logger.info(f'Finished fitting model')
        return loss_vals
