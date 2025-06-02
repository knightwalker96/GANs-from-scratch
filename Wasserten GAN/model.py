import torch
import torch.nn as nn
from torch.nn.modules import ConvTranspose2d

class Discriminator(nn.Module):
    def __init__(self, channel_img, features_d, use_instance_norm=False):
        super(Discriminator, self).__init__()
        norm_layer = nn.InstanceNorm2d if use_instance_norm else nn.BatchNorm2d
        self.disc = nn.Sequential(
            nn.Conv2d(channel_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1, norm_layer),
            self._block(features_d*2, features_d*4, 4, 2, 1, norm_layer),
            self._block(features_d*4, features_d*8, 4, 2, 1, norm_layer),
            nn.Conv2d(features_d*8, 1, 4, 2, 0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, norm_layer):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, use_instance_norm=False):
        super(Generator, self).__init__()
        norm_layer = nn.InstanceNorm2d if use_instance_norm else nn.BatchNorm2d
        self.gen = nn.Sequential(
            self._block(z_dim, features_g*16, 4, 1, 0, norm_layer),
            self._block(features_g*16, features_g*8, 4, 2, 1, norm_layer),
            self._block(features_g*8, features_g*4, 4, 2, 1, norm_layer),
            self._block(features_g*4, features_g*2, 4, 2, 1, norm_layer),
            nn.ConvTranspose2d(features_g*2, channels_img, 4, 2, 1),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, norm_layer):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            norm_layer(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 0.0, 0.02)