import torch
import torch.nn as nn
from torch.nn.modules import ConvTranspose2d

class Discriminator(nn.Module):
    def __init__(self, channel_img, features_d, num_classes, img_size, use_instance_norm=False):
        super(Discriminator, self).__init__()
        norm_layer = nn.InstanceNorm2d if use_instance_norm else nn.BatchNorm2d
        self.img_size = img_size
        self.disc = nn.Sequential(
            nn.Conv2d(channel_img + 1, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1, norm_layer),
            self._block(features_d*2, features_d*4, 4, 2, 1, norm_layer),
            self._block(features_d*4, features_d*8, 4, 2, 1, norm_layer),
            nn.Conv2d(features_d*8, 1, 4, 2, 0),
        )

        self.embed = nn.Embedding(num_classes , img_size * img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, norm_layer):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x, num_classes, labels):
        embed = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x , embed] , dim = 1)     ### N * C * img_size (H) * img_Size (W)
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, num_classes, img_size, embed_size, use_instance_norm=False):
        super(Generator, self).__init__()
        norm_layer = nn.InstanceNorm2d if use_instance_norm else nn.BatchNorm2d
        self.img_size = img_size
        self.embed_size = embed_size
        self.gen = nn.Sequential(
            self._block(z_dim + embed_size, features_g*16, 4, 1, 0, norm_layer),
            self._block(features_g*16, features_g*8, 4, 2, 1, norm_layer),
            self._block(features_g*8, features_g*4, 4, 2, 1, norm_layer),
            self._block(features_g*4, features_g*2, 4, 2, 1, norm_layer),
            nn.ConvTranspose2d(features_g*2, channels_img, 4, 2, 1),
            nn.Tanh()
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, norm_layer):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            norm_layer(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        # latent vector z: N * noise_dim * 1 * 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x , embedding] , dim = 1)
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 0.0, 0.02)