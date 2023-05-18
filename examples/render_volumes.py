import os
import yaml
import argparse
import numpy as np
import torch

from neural_canvas.models.inrf import INRF3D
from neural_canvas.runners import runner3d

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from pyvista.themes import DefaultTheme
from dask_image.imread import imread
import logging
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
import vtk


#def TF(volume):
#    grads = np.gradient(volume) 
#    grads = np.stack(grads, 0)
#    opacity = np.linalg.norm(grads[:3], axis=0).mean(-1, keepdims=True)  # [N, N, N]
#    return opacity

def TF(volume, grads=None, grad_dim=None, grad_reduction='norm', rgb_reduction='mean'):
    if grads is None:
        grads = np.gradient(volume)
    if grad_dim is not None:
        grads = grads[grad_dim]
    else:
        grads = grads[:3]
        if grad_reduction == 'norm':
            grads = np.linalg.norm(grads, axis=0)
        elif grad_reduction == 'mean':
            grads = grads.mean(0)
        elif grad_reduction == 'sum':  
            grads = grads.sum(0)

    if rgb_reduction == 'mean':
        opacity = grads.mean(-1, keepdims=True)
    elif rgb_reduction == 'sum':    
        opacity = grads.sum(-1, keepdims=True)
    elif rgb_reduction == 'norm':
        opacity = np.linalg.norm(grads, axis=-1, keepdims=True)
    return opacity


def render_options(volume):
    theme = DefaultTheme()
    theme.background = 'black'
    pv.global_theme.load_theme(theme)
    grid = pv.UniformGrid(dimensions=volume.shape[:-1])

    p = pv.Plotter(shape=(2, 6),
                   notebook=False,
                   multi_samples=32,
                   line_smoothing=True,
                   polygon_smoothing=True,)
    
    opacity = np.ones_like(volume[..., 0])[..., None]
    rgba = np.concatenate((volume, opacity), -1).reshape(-1, 4).astype(np.uint8)
    p.add_volume(grid,
                 scalars=rgba, 
                 ambient=1.0,
                 mapper="smart",
                 diffuse=.5,
                 blending='composite',
                 shade=False)
    p.add_text('Constant Opacity', font_size=10)
    grads = np.gradient(volume)  # [3, N, N, N, 3]
    grads = np.stack(grads, 0)
    p.subplot(0, 1)
    opacity = TF(volume, grads, grad_dim=None, grad_reduction='mean', rgb_reduction='norm')
    #opacity = np.linalg.norm(grads[:3].mean(0), axis=-1, keepdims=True)  # [N, N, N]
    rgba = np.concatenate((volume, opacity), -1).reshape(-1, 4).astype(np.uint8)
    p.add_volume(grid,
                 scalars=rgba, 
                 ambient=1.0,
                 mapper="smart",
                 diffuse=.5,
                 blending='composite',
                 shade=False)
    p.add_text('RGB Norm, Mean GradXYZ', font_size=10)

    p.subplot(0, 2)
    opacity = TF(volume, grads, grad_dim=0, grad_reduction=None, rgb_reduction='mean')
    #opacity = grads[0].mean(-1, keepdims=True)  # [N, N, N, 1]
    rgba = np.concatenate((volume, opacity), -1).reshape(-1, 4).astype(np.uint8)
    p.add_volume(grid,
                 scalars=rgba,
                 ambient=1.0,
                 mapper="smart",
                 diffuse=.5,
                 blending='composite',
                 shade=False)
    p.add_text('RGB Mean GradX', font_size=10)

    p.subplot(0, 3)
    opacity = TF(volume, grads, grad_dim=1, grad_reduction=None, rgb_reduction='mean')
    #opacity = grads[1].mean(-1, keepdims=True)
    rgba = np.concatenate((volume, opacity), -1).reshape(-1, 4).astype(np.uint8)
    p.add_volume(grid,
                 scalars=rgba,
                 ambient=1.0,
                 mapper="smart",
                 diffuse=.5,
                 blending='composite',
                 shade=False)
    p.add_text('RGB Mean GradY', font_size=10)

    p.subplot(0, 4)
    opacity = TF(volume, grads, grad_dim=2, grad_reduction=None, rgb_reduction='mean')
    #opacity = grads[2].mean(-1, keepdims=True)
    rgba = np.concatenate((volume, opacity), -1).reshape(-1, 4).astype(np.uint8)
    p.add_volume(grid,
                 scalars=rgba,
                 ambient=1.0,
                 mapper="smart",
                 diffuse=.5,
                 blending='composite',
                 shade=False)
    p.add_text('RGB Mean GradZ', font_size=10)

    p.subplot(0, 5)
    opacity = TF(volume, grads, grad_dim=None, grad_reduction='sum', rgb_reduction='mean')
    #opacity = grads[:3].mean(-1, keepdims=True).sum(0)
    rgba = np.concatenate((volume, opacity), -1).reshape(-1, 4).astype(np.uint8)
    p.add_volume(grid,
                 scalars=rgba,
                 ambient=1.0,
                 mapper="smart",
                 diffuse=.5,
                 blending='composite',
                 shade=False)
    p.add_text('RGB Mean, Sum Grad XYZ', font_size=10)

    p.subplot(1, 0)
    rgb = volume[..., 0].reshape(-1).astype(np.uint8)
    p.add_volume(grid,
                 scalars=rgb,
                 opacity='linear',
                 ambient=1.0,
                 mapper="smart",
                 diffuse=.5,
                 blending='composite',
                 shade=False)
    p.add_text('Linear TF', font_size=10)

    p.subplot(1, 1)
    opacity = TF(volume, grads, grad_dim=None, grad_reduction='norm', rgb_reduction='mean')    
    #opacity = np.linalg.norm(grads[:3], axis=0).mean(-1, keepdims=True)  # [N, N, N]
    rgba = np.concatenate((volume, opacity), -1).reshape(-1, 4).astype(np.uint8)
    p.add_volume(grid,
                 scalars=rgba,
                 ambient=1.0,
                 mapper="smart",
                 diffuse=.5,
                 blending='composite',
                 shade=False)
    p.add_text('RGB Mean, GradXYZ norm', font_size=10)

    p.subplot(1, 2)
    opacity = TF(volume, grads, grad_dim=0, grad_reduction=None, rgb_reduction='norm')
    #opacity = np.linalg.norm(grads[0], axis=-1, keepdims=True)
    rgba = np.concatenate((volume, opacity), -1).reshape(-1, 4).astype(np.uint8)
    p.add_volume(grid,
                 scalars=rgba,
                 ambient=1.0,
                 mapper="smart",
                 diffuse=.5,
                 blending='composite',
                 shade=False)
    p.add_text('RGB Norm, GradX', font_size=10)

    p.subplot(1, 3)
    opacity = TF(volume, grads, grad_dim=1, grad_reduction=None, rgb_reduction='norm')
    #opacity = np.linalg.norm(grads[1], axis=-1, keepdims=True)
    rgba = np.concatenate((volume, opacity), -1).reshape(-1, 4).astype(np.uint8)
    p.add_volume(grid,
                 scalars=rgba,
                 ambient=1.0,
                 mapper="smart",
                 diffuse=.5,
                 blending='composite',
                 shade=False)
    p.add_text('RGB Norm, GradY', font_size=10)

    p.subplot(1, 4)
    opacity = TF(volume, grads, grad_dim=2, grad_reduction=None, rgb_reduction='norm')
    #opacity = np.linalg.norm(grads[2], axis=-1, keepdims=True)
    rgba = np.concatenate((volume, opacity), -1).reshape(-1, 4).astype(np.uint8)
    p.add_volume(grid,
                 scalars=rgba,
                 ambient=1.0,
                 mapper="smart",
                 diffuse=.5,
                 blending='composite',
                 shade=False)
    p.add_text('RGB Norm, GradZ', font_size=10)
    p.subplot(1, 5)
    opacity = TF(volume, grads, grad_dim=None, grad_reduction='sum', rgb_reduction='norm')
    #opacity = np.linalg.norm(grads[:3].sum(0), axis=-1, keepdims=True)
    rgba = np.concatenate((volume, opacity), -1).reshape(-1, 4).astype(np.uint8)
    p.add_volume(grid,
                 scalars=rgba,
                 ambient=1.0,
                 mapper="fixed_point",
                 #mapper="smart",
                 diffuse=.5,
                 blending='composite',
                 shade=False)
    p.add_text('RGB Norm, Sum Grad XYZ', font_size=10)
    p.show()


def render(volume, path):
    theme = DefaultTheme()
    theme.background = 'black'
    pv.global_theme.load_theme(theme)
    grid = pv.UniformGrid(dimensions=volume.shape[:-1])
    
    p = pv.Plotter(notebook=False,
                   multi_samples=8,
                   line_smoothing=True,
                   polygon_smoothing=True,)
    opacity = TF(volume)
    rgba = np.concatenate((volume, opacity), -1).reshape(-1, 4).astype(np.uint8)
    p.add_volume(grid,
                 scalars=rgba,
                 ambient=1.0,
                 mapper="open_gl",#smart",
                 diffuse=.5,
                 blending='composite',
                 shade=False)
    save_fn = path[:-4]
    p.show()


def load_args(argv=None):
    parser = argparse.ArgumentParser(description='INRF3D-config')
    parser.add_argument('--conf', default=None, type=str, help='args config file')
    parser.add_argument('--latent_dim', default=8, type=int, help='latent space width')
    parser.add_argument('--latent_scale', default=1.0, type=float, help='mutiplier on z')
    parser.add_argument('--num_samples', default=1, type=int, help='images to generate')
    parser.add_argument('--x_dim', default=256, type=int, help='out image width')
    parser.add_argument('--y_dim', default=256, type=int, help='out image height')
    parser.add_argument('--z_dim', default=256, type=int, help='out image depth')
    parser.add_argument('--c_dim', default=3, type=int, help='channels')
    parser.add_argument('--mlp_layer_width', default=32, type=int, help='net width')    
    parser.add_argument('--activations', default='fixed', type=str,
        help='activation set for generator')
    parser.add_argument('--final_activation', default='sigmoid', type=str, help='last activation')
    parser.add_argument('--graph_topology', default='mlp', type=str,
        help='graph style to use for generator', choices=['mlp', 'WS'])
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--use_gpu', action='store_true', help='use gpu')
    parser.add_argument('--ws_graph_nodes', default=10, type=int, help='number of nodes in ws graph')
    parser.add_argument('--weight_init', default='normal', type=str, help='weight init scheme')
    parser.add_argument('--weight_init_mean', default=0.0, type=float, help='weight init mean')
    parser.add_argument('--weight_init_std', default=1.0, type=float, help='weight init std')
    parser.add_argument('--weight_init_max', default=2.0, type=float, help='weight init max')
    parser.add_argument('--weight_init_min', default=-2.0, type=float, help='weight init min')

    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--ignore_seed', action='store_true', help='ignore random seed, ' \
        'useful for running the same config multiple times without changing seed each run')
    parser.add_argument('--random_settings', action='store_true', help='ignore config, use random settings')
    parser.add_argument('--splits', default=1, type=int, help='number of splits for data')
    parser.add_argument('--output_dir', default='trial', type=str, help='output fn')
    parser.add_argument('--tmp_dir', default='trial', type=str, help='output fn')

    parser.add_argument('--colormaps', type=str, default=None, help='colormaps to save out',
        choices=['gray', 'hsv', 'lab', 'hls', 'luv'])

    # For Zoom videos
    parser.add_argument('--zoom_bounds', default=(.5, .5, .5), type=tuple, help='zoom in/out boundaries')
    parser.add_argument('--zoom_schedule', default='linear', type=str, help='zoom in/out pacing',
        choices=['linear', 'geometric', 'cosine', 'sigmoid', 'exp', 'log', 'sqrt', 'cbrt'])
    # For Pan videos
    parser.add_argument('--pan_bounds', default=(2, 2, 2), type=tuple, help='pan boundaries')
    parser.add_argument('--pan_schedule', default='linear', type=str, help='pan pacing',
        choices=['linear', 'geometric', 'cosine', 'sigmoid', 'exp', 'log', 'sqrt', 'cbrt'])
    # For regenerating and augmenting images
    parser.add_argument('--regen_image_path', type=str, default=None, help='path to image to regenerate')
    parser.add_argument('--regen_x_dim', type=int, default=512, help='width of output image')
    parser.add_argument('--regen_y_dim', type=int, default=512, help='height of output image')
    parser.add_argument('--regen_z_dim', type=int, default=512, help='height of output image')
    parser.add_argument('--regen_c_dim', type=int, default=3, help='channels in output image')
    parser.add_argument('--save_video_from_volumes', action='store_true', help='save video')

    args, _ = parser.parse_known_args(argv)
    if args.conf is not None:
        if os.path.isfile(args.conf):
            with open(args.conf, 'r') as f:
                defaults = yaml.safe_load(f)
            
            defaults = {k: v for k, v in defaults.items() if v is not None}
            parser.set_defaults(**defaults)
            args, _ = parser.parse_known_args(argv)
    
    return args


if __name__ == '__main__':
    args = load_args()
    if args.use_gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print (f'Using device: {device}')
    if args.random_settings:
        args.latent_dim = int(np.random.choice([8, 16, 32, 64]))
        args.latent_scale = float(np.random.choice([8, 16, 32]))
        args.mlp_layer_width = int(np.random.choice([16, 32]))
        args.activations = str(np.random.choice(['random']))
        args.graph_topology = str(np.random.choice(['mlp', 'WS']))
        args.ws_graph_nodes = int(np.random.choice([8, 16, 32]))
        args.weight_init = str(np.random.choice(['normal']))#, 'uniform']))
        args.weight_init_mean = 0.0 #float(np.random.choice([-3, -2, -1, 0, 1, 2, 3]))
        args.weight_init_std = float(np.random.choice([.85, 1.0, 1.15]))
        args.weight_init_min = float(np.random.choice([-3, -2, -1]))
        args.weight_init_max = float(np.random.choice([1, 2, 3]))
        print (f'Randomly selected settings: {args}')

    generator = INRF3D(latent_dim=args.latent_dim,
                       latent_scale=args.latent_scale,
                       output_shape=(args.x_dim, args.y_dim, args.z_dim, args.c_dim),
                       output_dir=args.output_dir,
                       tmp_dir=args.tmp_dir,
                       seed=args.seed if not args.ignore_seed else None,
                       device=device)
    
    generator.init_map_fn(mlp_layer_width=args.mlp_layer_width,
                          activations=args.activations,
                          final_activation=args.final_activation,
                          weight_init=args.weight_init,
                          graph_topology=args.graph_topology,
                          weight_init_mean=args.weight_init_mean,
                          weight_init_std=args.weight_init_std,
                          weight_init_min=args.weight_init_min,
                          weight_init_max=args.weight_init_max,)
    
    runner = runner3d.RunnerINRF3D(model=generator,
                          output_dir=args.output_dir,
                          save_verbose=False,
                          skip_blank_generations=True,
                          colormaps=args.colormaps)    
        
    volumes, _, paths = runner.run_volumes(num_samples=args.num_samples, splits=args.splits)
    render_options(volumes[0])
    render(volumes[0], paths[0])
