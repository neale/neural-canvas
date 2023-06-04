import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_canvas.losses.lpips_loss import LPIPS
from neural_canvas.models.discriminator_vqgan import NLayerDiscriminator
from pytorch_msssim import ssim, ms_ssim


def init_perceptual_loss(device):
    pips_loss = LPIPS().eval().to(device)
    return pips_loss


def init_discriminator_loss(device):
    # load VQGAN discriminator
    assert os.path.isfile('./last.ckpt'), 'VQGAN discriminator not found'
    discriminator = NLayerDiscriminator(3, 64, 2)
    d_state = torch.load('./last.ckpt')
    d_state = {k[19:]: v for k, v in d_state['state_dict'].items() if 'discriminator' in k}
    discriminator.load_state_dict(d_state)
    discriminator = discriminator.to(device).eval()
    return discriminator


def l1_loss(x, y):
    assert x.shape == y.shape, 'x and y must have the same shape'
    return F.l1_loss(x, y) 


def l2_loss(x, y):
    assert x.shape == y.shape, 'x and y must have the same shape'
    return F.mse_loss(x, y)


def perceptual_loss(x, y, fn):
    assert x.shape == y.shape, 'x and y must have the same shape'
    return fn(x, y)[0, 0, 0, 0]
            

def ssim_loss(x, y, multiscale=True):
    assert x.shape == y.shape, 'x and y must have the same shape'
    assert x.shape[1] == 3 or x.shape[1] == 1, f'x and y must have {3, 1} channels, got {x.shape[1]}'
    assert y.shape[1] == 3 or y.shape[1] == 1, f'x and y must have {3, 1} channels, got {y.shape[1]}'
    if multiscale:
        try:
            loss = ms_ssim(x, y, data_range=1, size_average=True)
        except AssertionError:
            loss = ssim(x, y, data_range=1, size_average=True)    
    else:
        loss = ssim(x, y, data_range=1, size_average=True)    
    return loss


def embedding_loss(z):
    return (z**2).mean()
            

def discriminator_loss(x, discriminator):
    logits_fake = discriminator(x)
    loss = logits_fake.mean()
    return loss


class LossModule(nn.Module):
    def __init__(self,
                 perceptual_alpha=0.0,
                 discriminator_alpha=0.0, 
                 l1_alpha=0.0,
                 l2_alpha=0.0,
                 ssim_alpha=0.0,
                 embedding_alpha=0.0,
                 device='cpu'):
        super(LossModule, self).__init__()
        self.use_perceptual_loss = perceptual_alpha > 0.0
        self.use_discriminator_loss = discriminator_alpha > 0.0
        self.use_l1_loss = l1_alpha > 0.0
        self.use_l2_loss = l2_alpha > 0.0
        self.use_ssim_loss = ssim_alpha > 0.0
        self.use_embedding_loss = embedding_alpha > 0.0

        self.perceptual_alpha = perceptual_alpha
        self.discriminator_alpha = discriminator_alpha
        self.l1_alpha = l1_alpha
        self.l2_alpha = l2_alpha
        self.ssim_alpha = ssim_alpha
        self.embedding_alpha = embedding_alpha

        if self.use_perceptual_loss:
            self.perceptual_loss_fn = init_perceptual_loss(device)
        if self.use_discriminator_loss:
            self.discriminator_loss_fn = init_discriminator_loss(device)

    def forward(self, x, y):
        loss = 0.0
        if self.use_discriminator_loss:
            loss += self.discriminator_alpha * self.discriminator_loss_fn(
                x, self.discriminator_loss_fn)
        if self.use_perceptual_loss:
            loss += self.perceptual_alpha * self.perceptual_loss_fn(
                x, y, self.perceptual_loss_fn)
        if self.use_l1_loss:
            loss += self.l1_alpha * l1_loss(x, y)
        if self.use_l2_loss:
            loss += self.l2_alpha * l2_loss(x, y)
        if self.use_ssim_loss:
            loss += self.ssim_alpha * ssim_loss(x, y)
        if self.use_embedding_loss:
            loss += self.embedding_alpha * embedding_loss(x)
        return loss