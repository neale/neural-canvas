import os
import yaml
import glob
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib.pyplot').setLevel(logging.ERROR)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.ERROR)

from neural_canvas.models.inrf import INRF2D
from neural_canvas.runners import runner2d
from neural_canvas.utils import utils


def load_args(argv=None):
    parser = argparse.ArgumentParser(description='INRF2D-config')
    parser.add_argument('--conf', default=None, type=str, help='args config file')
    parser.add_argument('--latent_dim', default=8, type=int, help='latent space width')
    parser.add_argument('--latent_scale', default=1.0, type=float, help='mutiplier on z')
    parser.add_argument('--num_samples', default=1, type=int, help='images to generate')
    parser.add_argument('--x_dim', default=256, type=int, help='out image width')
    parser.add_argument('--y_dim', default=256, type=int, help='out image height')
    parser.add_argument('--c_dim', default=3, type=int, help='channels')
    parser.add_argument('--activations', default='fixed', type=str,
        help='activation set for generator')
    parser.add_argument('--final_activation', default='tanh', type=str, help='last activation',
        choices=['sigmoid', 'tanh', None])
    parser.add_argument('--graph_topology', default='conv_fixed', type=str,
        help='graph style to use for generator', choices=['conv_fixed', 'WS'])
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--mlp_layer_width', default=32, type=int, help='net width')    
    parser.add_argument('--conv_feature_map_size', default=64, type=int, help='conv net width')
    parser.add_argument('--use_gpu', action='store_true', help='use gpu')
    parser.add_argument('--ws_graph_nodes', default=10, type=int, help='number of nodes in ws graph')
    parser.add_argument('--weight_init', default=None, type=str, help='weight init scheme')
    parser.add_argument('--weight_init_mean', default=0.0, type=float, help='weight init mean')
    parser.add_argument('--weight_init_std', default=1.0, type=float, help='weight init std')
    parser.add_argument('--weight_init_max', default=2.0, type=float, help='weight init max')
    parser.add_argument('--weight_init_min', default=-2.0, type=float, help='weight init min')
    parser.add_argument('--num_freqs_encoding', default=5, type=int, help='number of (pi)2^i frequencies' \
        'to use for positional encodings')
    parser.add_argument('--fourier_encoding', action='store_true', help='use fourier input encodings')

    parser.add_argument('--discriminator_loss_weight', default=0.0, type=float, help='discriminator loss weight')
    parser.add_argument('--l1_loss_weight', default=1.0, type=float, help='l1 loss weight')
    parser.add_argument('--l2_loss_weight', default=0.0, type=float, help='l2 loss weight')
    parser.add_argument('--embedding_loss_weight', default=0.0, type=float, help='embedding loss weight')
    parser.add_argument('--ssim_loss_weight', default=0.0, type=float, help='ssim loss weight')
    parser.add_argument('--perceptual_loss_weight', default=0.0, type=float, help='perceptual loss weight')

    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--ignore_seed', action='store_true', help='ignore random seed, ' \
        'useful for running the same config multiple times without changing seed each run')

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.00005, type=float, help='learning rate decay')
    parser.add_argument('--num_epochs', default=50, type=int, help='number of epochs to train for')
    parser.add_argument('--num_iters_per_epoch', default=50, type=int, help='number of iters per epoch')
    parser.add_argument('--input_dir', default=None, type=str, help='directory of images to fit')
    parser.add_argument('--input_file', default=None, type=str, help='target image to fit')
    parser.add_argument('--output_dir', default='trial', type=str, help='output fn')
    parser.add_argument('--tmp_dir', default='trial', type=str, help='output fn')


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
    print (f'Using device: {device}')


    assert args.input_dir is not None or args.input_file is not None, 'Must specify input file or directory'
    if args.input_dir is not None:
        if os.path.isdir(args.input_dir):
            image_paths = glob.glob(os.path.join(args.input_dir, '*.png'))
            image_paths += glob.glob(os.path.join(args.input_dir, '*.jpg'))
        else:
            raise ValueError(f'Invalid input directory, got {args.input_dir}')
    elif args.input_file is not None:
        if os.path.isfile(args.input_file):
            image_paths = [args.input_file]
        else:
            raise ValueError(f'Invalid input file, got {args.input_file}')

    for path in image_paths:
        target = utils.load_image_as_tensor(path, args.output_dir)

        generator = INRF2D(latent_dim=args.latent_dim,
                           latent_scale=args.latent_scale,
                           output_shape=(target.shape[2], target.shape[3], target.shape[1]),
                           output_dir=args.output_dir,
                           tmp_dir=args.tmp_dir,
                           seed=args.seed if not args.ignore_seed else None,
                           device=device)

        if args.fourier_encoding:
            input_freqs_encoding = 2 * args.num_freqs_encoding
        else:
            input_freqs_encoding = 1

        generator.init_map_fn(mlp_layer_width=args.mlp_layer_width,
                              conv_feature_map_size=args.conv_feature_map_size,
                              input_encoding_dim=input_freqs_encoding,
                              activations=args.activations,
                              final_activation=args.final_activation,
                              weight_init=args.weight_init,
                              num_graph_nodes=args.ws_graph_nodes,
                              graph_topology=args.graph_topology,
                              weight_init_mean=args.weight_init_mean,
                              weight_init_std=args.weight_init_std,
                              weight_init_min=args.weight_init_min,
                              weight_init_max=args.weight_init_max,
                              num_fourier_freqs=args.num_freqs_encoding,)

        
        runner = runner2d.RunnerINRF2D(model=generator,
                                       output_dir=args.output_dir,
                                       save_verbose=False)
        loss_vals = runner.fit(target,
                               output_shape=(args.x_dim, args.y_dim, args.c_dim),
                               loss_weights = {
                                   'perceptual_alpha': args.perceptual_loss_weight,
                                   'l1_alpha': args.l1_loss_weight,
                                   'l2_alpha': args.l2_loss_weight,
                                   'embedding_alpha': args.embedding_loss_weight,
                                   'discriminator_alpha': args.discriminator_loss_weight,
                                   'ssim_alpha': args.ssim_loss_weight
                               },
                               num_epochs=args.num_epochs,
                               num_iters_per_epoch=args.num_iters_per_epoch,
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               device=device,)
        
        img_name = path.split('/')[-1][:-4]
        plt.plot(np.arange(len(loss_vals)), loss_vals, label='INRF_loss')
        plt.title(f'INRF fit loss on {path}')
        plt.xlabel(f'Optimization step x {args.num_iters_per_epoch}')
        plt.ylabel(f'Sum of Losses')
        plt.grid()
        plt.legend(loc='best')
        plt.savefig(os.path.join(args.output_dir, img_name+'_loss.png'))
        plt.close()