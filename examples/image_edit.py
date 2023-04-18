import os
import yaml
import argparse

import numpy as np
import torch

from neural_canvas.utils import schedulers
from neural_canvas.models.inrf import INRF2D
from neural_canvas.runners import runner2d


def load_args(argv=None):
    parser = argparse.ArgumentParser(description='INRF2D-edit-config')
    parser.add_argument('--conf', default=None, type=str, help='args config file')
    parser.add_argument('--latent_dim', default=8, type=int, help='latent space width')
    parser.add_argument('--latent_scale', default=1.0, type=float, help='mutiplier on z')
    parser.add_argument('--num_samples', default=1, type=int, help='images to generate')
    parser.add_argument('--x_dim', default=256, type=int, help='out image width')
    parser.add_argument('--y_dim', default=256, type=int, help='out image height')
    parser.add_argument('--c_dim', default=4, type=int, help='channels')
    parser.add_argument('--mlp_layer_width', default=32, type=int, help='net width')    
    parser.add_argument('--activations', default='fixed', type=str,
        help='activation set for generator')
    parser.add_argument('--final_activation', default='sigmoid', type=str, help='last activation')
    parser.add_argument('--graph_topology', default='mlp_fixed', type=str,
        help='graph style to use for generator', choices=['mlp_fixed', 'WS'])
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

    parser.add_argument('--output_dir', default='trial', type=str, help='output fn')
    parser.add_argument('--tmp_dir', default='trial', type=str, help='output fn')

    parser.add_argument('--colormaps', type=str, default=None, help='colormaps to save out',
        choices=['gray', 'hsv', 'lab', 'hls', 'luv'])

    # For Zoom videos
    parser.add_argument('--save_zoom_stack', action='store_true', help='save zoomed in/out frames')
    parser.add_argument('--zoom_bounds', default=(.5, .5), type=tuple, help='zoom in/out boundaries')
    parser.add_argument('--zoom_scheduler', default=None, type=str, help='zoom in/out pacing',
        choices=['linear', 'geometric', 'cosine', 'sigmoid', 'exp', 'log', 'sqrt'])
    # For Pan videos
    parser.add_argument('--pan_bounds', default=(2, 2), type=tuple, help='pan boundaries')
    parser.add_argument('--pan_scheduler', default=None, type=str, help='pan pacing',
        choices=['linear', 'geometric', 'cosine', 'sigmoid', 'exp', 'log', 'sqrt'])
    # For regenerating and augmenting images
    parser.add_argument('--regen_image_path', type=str, default=None, help='path to image to regenerate')
    parser.add_argument('--regen_x_dim', type=int, default=512, help='width of output image')
    parser.add_argument('--regen_y_dim', type=int, default=512, help='height of output image')
    parser.add_argument('--regen_c_dim', type=int, default=3, help='channels in output image')
    parser.add_argument('--save_video_from_frames', action='store_true', help='save video')

    args, _ = parser.parse_known_args(argv)
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
    if args.zoom_scheduler is not None:
        print (f'Using zoom scheduler: {args.zoom_scheduler}')
        zoom_schedule = getattr(schedulers, args.zoom_scheduler)(
            *args.zoom_bounds, args.num_samples)
        zoom_schedule = torch.stack([zoom_schedule, zoom_schedule], -1)
        print (zoom_schedule)

    else:
        zoom_schedule = None
    if args.pan_scheduler is not None:
        print (f'Using pan scheduler: {args.pan_scheduler}')
        pan_schedule = getattr(schedulers, args.pan_scheduler)(
            *args.pan_bounds, args.num_samples)
        pan_schedule = torch.stack([pan_schedule, pan_schedule], -1)
    else:
        pan_schedule = None
    
    if args.regen_image_path is not None:
        runner = runner2d.RunnerINRF2D(output_dir=args.output_dir, colormaps=args.colormaps)
        runner.regen_frames(path=args.regen_image_path,
                            output_shape=(args.regen_x_dim, args.regen_y_dim, args.regen_c_dim),
                            num_samples=args.num_samples,
                            splits=1,
                            zoom_schedule=zoom_schedule,
                            pan_schedule=pan_schedule,
                            save_video=args.save_video_from_frames)