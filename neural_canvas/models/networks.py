import torch
import torch.nn as nn
import torch.nn.functional as F


class ZSTransform(nn.Module):
    def __init__(self, z_dim):
        super(ZSTransform, self).__init__()
        self.linear1 = nn.Linear(z_dim, z_dim)
        self.linear2 = nn.Linear(z_dim, z_dim)
        self.linear_mean = nn.Linear(z_dim, 1)
        self.linear_std = nn.Linear(z_dim, 1)

    def forward(self, z):
        z = F.silu(self.linear1(z))
        z = F.silu(self.linear2(z))
        mean = self.linear_mean(z)
        std = self.linear_std(z)
        return mean, std


class ZDTransform(nn.Module):
    def __init__(self, z_dim):
        super(ZDTransform, self).__init__()
        self.linear1 = nn.Linear(z_dim, z_dim)
        self.linear2 = nn.Linear(z_dim, z_dim)
        self.linear3 = nn.Linear(z_dim, z_dim)

    def forward(self, z):
        z = F.silu(self.linear1(z))
        z = F.silu(self.linear2(z))
        z = self.linear3(z)
        return z
    

class LinBnReLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinBnReLU, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = F.gelu(x)
        return x


class LinDecoder(nn.Module):
    def __init__(self, z_dim):
        super(LinDecoder, self).__init__()
        self.block1 = LinBnReLU(128, 512)
        self.block2 = LinBnReLU(512, 2048)
        self.block3 = LinBnReLU(2048, 256*256*3)
        self.fc_input = nn.Linear(z_dim, 128)

    def forward(self, z):
        z = F.gleu(self.fc_input(z))
        out = self.block1(z)
        out = self.block2(out)
        out = self.block3(out)
        out = torch.tanh(out)
        return out


class ConvTBnReLU(nn.Module):
    def __init__(self, filters_in, filters_out, padding=0, output_padding=0, upsample=1):
        super(ConvTBnReLU, self).__init__()
        self.conv = nn.ConvTranspose2d(filters_in, filters_out, kernel_size=3,
            stride=2, padding=padding, output_padding=output_padding)
        self.bn = nn.BatchNorm2d(filters_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class ConvDecoder(nn.Module):  # only works for 256 sized square images
    def __init__(self, z_dim):
        super(ConvDecoder, self).__init__()
        nf = 2
        self.block1 = ConvTBnReLU(256, nf*8, padding=1, output_padding=1, upsample=2)
        self.block2 = ConvTBnReLU(nf*8, nf*8, padding=1, output_padding=1, upsample=2)
        self.block3 = ConvTBnReLU(nf*8, nf*4, padding=1, output_padding=1, upsample=2)
        self.block4 = ConvTBnReLU(nf*4, nf*4, padding=1, output_padding=1, upsample=2)
        self.block5 = ConvTBnReLU(nf*4, nf*2, padding=1, output_padding=1, upsample=2)
        self.block6 = ConvTBnReLU(nf*2, nf, padding=1, output_padding=0, upsample=2)
        self.block7 = ConvTBnReLU(nf, 3, padding=0, output_padding=1, upsample=1)
        self.fc_input = nn.Linear(z_dim, 256*2*2)

    def forward(self, z):
        z = F.silu(self.fc_input(z))
        out = z.view(-1, 256, 2, 2)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = torch.tanh(out)
        return out