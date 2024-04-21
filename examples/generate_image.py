import os
import yaml
import argparse
import numpy as np
import torch

from neural_canvas.models.inrf import INRF2D
from neural_canvas.runners import runner2d
  

def load_args(argv=None):
    parser = argparse.ArgumentParser(description='INRF2D-config')
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
    parser.add_argument('--final_activation', default=None, type=str, help='last activation',
        choices=['sigmoid', 'tanh', None])
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
    for _ in range(args.num_samples):
        if args.random_settings:
            args.latent_dim = int(np.random.choice([8, 16, 32, 64]))
            args.latent_scale = float(np.random.choice([.1, .5, 1, 4, 8, 16, 32, 64]))
            args.mlp_layer_width = int(np.random.choice([16, 32, 64]))
            args.activations = str(np.random.choice(['fixed', 'random']))
            args.graph_topology = str(np.random.choice(['simple', 'mlp', 'WS']))
            args.ws_graph_nodes = int(np.random.choice([8, 16, 32, 64]))
            args.final_activation = np.random.choice(['sigmoid', 'tanh', None])
            args.weight_init = str(np.random.choice(['normal', 'uniform']))
            args.weight_init_mean = float(np.random.choice([-3, -2, -1, 0, 1, 2, 3]))
            args.weight_init_std = float(np.random.choice([.5, .85, 1.0, 1.15]))
            args.weight_init_min = float(np.random.choice([-3, -2, -1]))
            args.weight_init_max = float(np.random.choice([1, 2, 3]))
            print (f'Randomly selected settings: {args}')

        generator = INRF2D(latent_dim=args.latent_dim,
                        latent_scale=args.latent_scale,
                        output_shape=(args.x_dim, args.y_dim, args.c_dim),
                        output_dir=args.output_dir,
                        tmp_dir=args.tmp_dir,
                        seed=args.seed if not args.ignore_seed else None,
                        device=device)
        
        generator.init_map_fn(mlp_layer_width=args.mlp_layer_width,
                            activations=args.activations,
                            final_activation=args.final_activation,
                            weight_init=args.weight_init,
                            num_graph_nodes=args.ws_graph_nodes,
                            graph_topology=args.graph_topology,
                            weight_init_mean=args.weight_init_mean,
                            weight_init_std=args.weight_init_std,
                            weight_init_min=args.weight_init_min,
                            weight_init_max=args.weight_init_max,)
        
        runner = runner2d.RunnerINRF2D(model=generator,
                            output_dir=args.output_dir,
                            save_verbose=False,
                            skip_blank_generations=True,
                            colormaps=args.colormaps)

        runner.run_frames(num_samples=1, splits=1)