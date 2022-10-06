import torch
from torch import nn

class TimePoseFunction(nn.Module):
    def __init__(self, hparams):
        super(TimePoseFunction, self).__init__()
        nets = []
        n_channels = hparams.n_channels
        for i in range(hparams.n_layers):
            if i == 0:
                layer = nn.Linear(1, n_channels)
            elif i in hparams.skip_connections:
                layer = nn.Linear(n_channels, n_channels)
            layer = nn.Sequential(layer, nn.LeakyReLU(inplace=True, negative_slope=0.01))
            nets.append(layer)
        
        self.encoder = nn.ModuleList(nets)
        self.quaternion_decoder = nn.Sequential(nn.Linear(n_channels, 4), nn.Sigmoid())
        self.translation_decoder = nn.Linear(n_channels, 3)
        
    def forward(self, t):
        embedding = self.encoder(t)
        return {
            "rotation_quaternion": self.quaternion_decoder(embedding),
            "translation": self.translation_decoder(t)
        }
