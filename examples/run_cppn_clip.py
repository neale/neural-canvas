import os
import gc
import sys
import tqdm
import glob
import shutil
import argparse
import logging
import tifffile

import utils
import clip

import cv2
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from sklearn.datasets import make_blobs
from torch.utils.data import Subset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# Because imageio uses the root logger instead of warnings package

from networks.mlp_mixer import MLPMixer
from primitives import p_gmm, p_squares_right, p_squares_left, p_grad_img
from maps import MapRandomGraph, MapRandomAct, Map, plot_graph, ConvDecoder, LinDecoder, MapConv

from losses.lpips_loss import LPIPS
from discriminator import NLayerDiscriminator

from losses.clip_utils import load_clip, MakeCutouts, spherical_dist_loss, tv_loss, parse_prompt, fetch, range_loss


logging.getLogger().setLevel(logging.ERROR)


class CPPN(object):
    """initializes a CPPN"""
    def __init__(self,
                 noise_dim=4,
                 noise_scale=10,
                 n_samples=6,
                 x_dim=512,
                 y_dim=512,
                 c_dim=3,
                 layer_width=4,
                 patch_size=4,
                 batch_size=1,
                 weight_init_mean=0.,
                 weight_init_std=1.0,
                 output_dir='.',
                 graph_nodes=10,
                 seed_gen=987654321, # 123456789,
                 seed=None,
                 use_conv=False,
                 use_linear=False,
                 use_mixer=False,
                 clip_loss=False,
                 device='cpu',
                 graph=None):

        self.noise_dim = noise_dim
        self.n_samples = n_samples
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.c_dim = c_dim 
        self.noise_scale = noise_scale
        self.weight_init_mean = weight_init_mean
        self.weight_init_std = weight_init_std
        self.layer_width = layer_width
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.output_dir = output_dir

        self.graph_nodes = graph_nodes
        self.graph = graph
        
        self.map_fn = None
        self.use_conv = use_conv
        self.conv_cppn = False
        self.use_linear = use_linear
        self.use_mixer = use_mixer
        self.clip_loss = clip_loss
        self.device = device
        self.logger = logging.getLogger('CPPNlogger')
        self.logger.setLevel(logging.INFO)

        self.seed = seed
        self.seed_gen = seed_gen
        self.init_random_seed(seed=seed)
        self._init_paths()
        
    def init_random_seed(self, seed=None):
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
        os.makedirs(self.output_dir, exist_ok=True)

    def _init_weights(self, model, layer_i, mul):
        for i, layer in enumerate(model.modules()):
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight.data,
                                self.weight_init_mean,
                                self.weight_init_std)
            if isinstance(layer, nn.Conv2d):
                pass
                #nn.init.normal_(layer.weight.data, 0, .01)

                if layer.out_channels == 3:
                    pass
                    #nn.init.zeros_(layer.weight.data)   

                else:
                    pass
                    #std = (1./layer.in_channels)**.5
                    #nn.init.normal_(layer.weight.data, 0, np.sqrt(1./layer.in_channels))
                #nn.init.zeros_(layer.bias.data)
        return model

    def init_map_fn(self,
                    seed=None,
                    activations='fixed',
                    graph_topology='fixed',
                    layer_i=0,
                    mul=1.0,
                    graph=None,
                    activation_set='large'):
        if self.use_conv:
            map_fn = ConvDecoder(self.noise_dim)
        elif self.use_linear:
            map_fn = LinDecoder(self.noise_dim)
        elif self.use_mixer:
            map_fn = MLPMixer(
                image_size=self.x_dim,
                channels=self.c_dim,
                patch_size=self.patch_size,
                dim=self.layer_width,
                depth=4,
                randomize_act=(activations!='fixed'),
                num_classes=self.x_dim*self.y_dim*self.c_dim)
        elif graph_topology == 'fixed' and activations == 'fixed':
            map_fn = Map(
                self.noise_dim, self.c_dim, self.layer_width, self.noise_scale)
        elif graph_topology == 'WS':
            map_fn = MapRandomGraph(
                self.noise_dim, self.c_dim, self.layer_width, self.noise_scale, self.graph_nodes,
                graph, activation_set, activations=activations)
        elif graph_topology == 'fixed' and activations == 'permute':
            map_fn = MapRandomAct(
                self.noise_dim, self.c_dim, self.layer_width, self.noise_scale)
        elif graph_topology == 'conv_fixed':
            self.conv_cppn = True
            map_fn = MapConv(
                self.noise_dim, self.c_dim, self.layer_width, self.noise_scale, clip_loss=self.clip_loss)
        else:
            raise NotImplementedError

        self.map_fn = self._init_weights(map_fn, layer_i, mul)
        self.map_fn = self.map_fn.to(self.device)

    def init_inputs(self, batch_size=None, x_dim=None, y_dim=None):
        if batch_size is None:
            batch_size = self.batch_size
        if x_dim is None:
            x_dim = self.x_dim
        if y_dim is None:
            y_dim = self.y_dim
        inputs = torch.ones(batch_size, 1, self.noise_dim).uniform_(-2., 2.)
        inputs = inputs.to(self.device)
        inputs = inputs.reshape(batch_size, 1, self.noise_dim)
        one_vec = torch.ones(x_dim*y_dim, 1).float().to(self.device)
        inputs_scaled = inputs * one_vec * self.noise_scale
        return inputs_scaled.unsqueeze(0).float()

    def _coordinates(self, x_dim, y_dim, batch_size, zoom, pan, as_mat=False):
        xpan, ypan = pan
        xzoom, yzoom = zoom
        x_dim, y_dim, scale = x_dim, y_dim, self.noise_scale
        n_pts = x_dim * y_dim
        x_range = scale*(np.arange(x_dim)-(x_dim-1)/xpan)/(x_dim-1)/xzoom
        y_range = scale*(np.arange(y_dim)-(y_dim-1)/ypan)/(y_dim-1)/yzoom
        x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
        y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
        r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
        if as_mat:
            return x_mat, y_mat, r_mat, n_pts
        x_vec = np.tile(x_mat.flatten(), batch_size).reshape(batch_size*n_pts, -1)
        y_vec = np.tile(y_mat.flatten(), batch_size).reshape(batch_size*n_pts, -1)
        r_vec = np.tile(r_mat.flatten(), batch_size).reshape(batch_size*n_pts, -1)

        x_vec = torch.from_numpy(x_vec).float().to(self.device)
        y_vec = torch.from_numpy(y_vec).float().to(self.device)
        r_vec = torch.from_numpy(r_vec).float().to(self.device)

        return x_vec, y_vec, r_vec, n_pts


def load_target_data(args, device):
    target = cv2.cvtColor(cv2.imread(args.target_img_path), cv2.COLOR_BGR2RGB)
    target = target / 255.
    target = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).float().to(device)
    target_fn = f'{args.output_dir}/target'
    utils.write_image(path=target_fn,
        img=target.permute(0, 2, 3, 1)[0].cpu().numpy()*255, suffix='jpg')
    return target



def run_cppn(args, cppn, debug=False):
    suff = f'z-{cppn.noise_dim}_scale-{cppn.noise_scale}_cdim-{cppn.c_dim}' \
           f'_net-{cppn.layer_width}_wmean-{cppn.weight_init_mean}_wstd-{cppn.weight_init_std}'
    # load target data
    if args.target_img_path == 'random':
        init_t = torch.rand(1, cppn.c_dim, cppn.x_dim, cppn.y_dim).to(cppn.device).requires_grad_(True)
    elif args.target_img_path is not None:
        init_t = load_target_data(args, cppn.device)
        print ('Using: ', cppn.device, ' target: ', init_t.shape)
    else:
        init_t = None
    
    optim_map = torch.optim.AdamW(cppn.map_fn.parameters(), 
                                  lr=1e-3, weight_decay=1e-5,
                                  betas=(.9, .999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optim_map, T_0=100, T_mult=2)

    # load LPIPS loss
    if args.perceptual_loss:
        pips_loss = LPIPS().eval().to(cppn.device)
    # load VQGAN discriminator
    if args.discriminator_loss:
        discriminator = NLayerDiscriminator(3, 64, 2)
        d_state = torch.load('./last.ckpt')
        d_state = {k[19:]: v for k, v in d_state['state_dict'].items() if 'discriminator' in k}
        discriminator.load_state_dict(d_state)
        discriminator = discriminator.to(cppn.device).eval()
    # Load CLIP loss
    if args.clip_loss:
        #assert args.perceptual_loss, 'CLIP loss requires perceptual loss'
        cutn = 16
        clip_model, clip_normalizer = load_clip(cppn.device)
        clip_size = clip_model.visual.input_resolution
        make_cutouts = MakeCutouts(clip_size, cutn=16)
        txt, weight = parse_prompt(args.clip_prompt)
        text_embedding = clip_model.encode_text(clip.tokenize(txt).to(cppn.device)).float()

    # load X, Y, R inputs to map function
    x_vec, y_vec, r_vec, n_pts = cppn._coordinates(
        cppn.x_dim, cppn.y_dim, 1, zoom=(.5,.5), pan=(2,2), as_mat=True)
    x_vec = torch.from_numpy(x_vec).to(cppn.device).float()
    y_vec = torch.from_numpy(y_vec).to(cppn.device).float() 
    r_vec = torch.from_numpy(r_vec).to(cppn.device).float()

    x2, y2, r2, _ = cppn._coordinates(
        1024, 1024, 1, zoom=(.25,.25), pan=(2,2), as_mat=True)
    x2 = torch.from_numpy(x2).to('cpu').float()
    y2 = torch.from_numpy(y2).to('cpu').float()
    r2 = torch.from_numpy(r2).to('cpu').float()

    losses = []
    loader = tqdm.tqdm(range(int(1e5)))
    for iteration in loader:
        optim_map.zero_grad()
        z = cppn.init_inputs()
        z_i = z.to(cppn.device)
        sample = cppn.map_fn(x_vec, y_vec, r_vec, z_i)
        loss = 0.
        clip_in = clip_normalizer(make_cutouts(sample.add(1).div(2)))
        image_embedding = clip_model.encode_image(clip_in).float() 
        loss_sph = spherical_dist_loss(image_embedding, text_embedding.unsqueeze(0)) 
        loss_range = range_loss(sample)# * 50 
        loss_tv = tv_loss(sample)# * 150
        loss = (loss_sph + loss_range + loss_tv).sum()

        loss.backward()
        optim_map.step()
        #scheduler.step(iteration)

        loader.set_description(f'Loss: {loss.detach().cpu().item():.4f}')

        if iteration % 100 == 0:
            save_fn = f'{cppn.output_dir}/iter{iteration}'
            sample = sample.add(1).div(2)
            #init_out = (init_t + 1) / 2.
            #img = init_out[0].permute(1, 2, 0).detach().cpu().numpy()*255
            img = sample[0].permute(1, 2, 0).detach().cpu().numpy()*255
            img = img.astype(np.uint8)
            utils.write_image(path=save_fn, img=img, suffix='jpg')
        """
        if iteration % 3000 == 0:
            save_fn = f'{cppn.output_dir}/iter{iteration}_big'
            z2 = cppn.init_inputs(x_dim=1024, y_dim=1024).to('cpu')
            cppn.map_fn = cppn.map_fn.to('cpu')
            sample = cppn.map_fn(x2, y2, r2, z2)[0] 
            sample = (sample + 1) / 2. 
            sample = sample.permute(1, 2, 0).detach().cpu().numpy()*255.
            utils.write_image(path=save_fn, img=sample, suffix='jpg')
            cppn.map_fn = cppn.map_fn.to(cppn.device)
        """
    py_files = glob.glob('*.py')
    assert len(py_files) > 0
    for fn in py_files:
        shutil.copy(fn, os.path.join(args.output_dir, fn))
    sys.exit(0)

def load_args():

    parser = argparse.ArgumentParser(description='cppn-pytorch')
    parser.add_argument('--noise_dim', default=2, type=int, help='latent space width')
    parser.add_argument('--n_samples', default=1, type=int, help='images to generate')
    parser.add_argument('--x_dim', default=256, type=int, help='out image width')
    parser.add_argument('--y_dim', default=256, type=int, help='out image height')
    parser.add_argument('--c_dim', default=3, type=int, help='channels')
    parser.add_argument('--layer_width', default=16, type=int, help='net width')
    parser.add_argument('--noise_scale', default=10, type=float, help='mutiplier on z')
    parser.add_argument('--patch_size', default=4, type=int, help='net width')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--graph_nodes', default=10, type=int, help='number of graph_nodes in graph')
    parser.add_argument('--output_dir', default='trial', type=str, help='output fn')
    parser.add_argument('--target_img_path', default=None, type=str, help='image to match')
    parser.add_argument('--optimize_deterministic_input', action='store_true')
    parser.add_argument('--optimize_stochastic_input', action='store_true')
    parser.add_argument('--optimize_map', action='store_true')
    parser.add_argument('--perceptual_loss', action='store_true')
    parser.add_argument('--ssim_loss', action='store_true')
    parser.add_argument('--l1_loss', action='store_true')
    parser.add_argument('--l2_loss', action='store_true')
    parser.add_argument('--discriminator_loss', action='store_true')
    parser.add_argument('--clip_loss', action='store_true')
    parser.add_argument('--clip_prompt', type=str, default='')
    parser.add_argument('--embedding_loss', action='store_true')
    parser.add_argument('--no_tiff', action='store_true', help='save tiff metadata')
    parser.add_argument('--activations', default='fixed', type=str, help='')
    parser.add_argument('--graph_topology', default='fixed', type=str, help='')
    parser.add_argument('--use_gpu', action='store_true', help='use GPU')
    parser.add_argument('--use_conv', action='store_true', help='use conv generator')
    parser.add_argument('--use_linear', action='store_true', help='use linear generator')
    parser.add_argument('--use_mixer', action='store_true', help='use linear generator')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = load_args()
    if args.use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    for _ in range(200):
        cppn = CPPN(noise_dim=args.noise_dim,
                    n_samples=args.n_samples,
                    x_dim=args.x_dim,
                    y_dim=args.y_dim,
                    c_dim=args.c_dim,
                    noise_scale=args.noise_scale,
                    layer_width=args.layer_width,
                    batch_size=args.batch_size,
                    patch_size=args.patch_size,
                    output_dir=args.output_dir,
                    device=device,
                    use_conv=args.use_conv,
                    use_linear=args.use_linear,
                    use_mixer=args.use_mixer,
                    clip_loss=args.clip_loss,
                    graph_nodes=args.graph_nodes)
        cppn.init_map_fn(activations=args.activations,
                        graph_topology=args.graph_topology)
        run_cppn(args, cppn, debug=False)
