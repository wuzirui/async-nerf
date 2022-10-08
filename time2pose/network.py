from typing import no_type_check
import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, num_freqs: int, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super(Embedding, self).__init__()

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, num_freqs - 1, num_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (num_freqs - 1), num_freqs)

    def get_out_channels(self, d_in=1):
        return (len(self.freq_bands) * 2 + 1) * d_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for freq in self.freq_bands:
            out += [torch.sin(freq * x), torch.cos(freq * x)]

        return torch.cat(out, -1)


class TimePoseFunction(nn.Module):
    def __init__(self, hparams):
        super(TimePoseFunction, self).__init__()
        nets = []
        n_channels = hparams.n_channels
        self.skip = hparams.skip_connections
        self.embedding = Embedding(num_freqs=hparams.t_embedding_freq)
        self.input_dir = self.embedding.get_out_channels(d_in=1)
        for i in range(hparams.n_layers):
            if i == 0:
                layer = nn.Linear(self.input_dir, n_channels)
            elif i in self.skip:
                layer = nn.Linear(self.input_dir + n_channels, n_channels)
            else:
                layer = nn.Linear(n_channels, n_channels)
            layer = nn.Sequential(
                layer, 
                nn.BatchNorm1d(n_channels),
                nn.Softplus()
            )
            nets.append(layer)
        
        self.nets= nn.ModuleList(nets)
        self.rotation_output = nn.Linear(n_channels, 4)
        self.translation_output = nn.Linear(n_channels, 3)
        
        
    def forward(self, t):
        emb = self.embedding(t)
        x = emb
        for i, net in enumerate(self.nets):
            if i in self.skip:
                x = torch.concat([emb, x], dim=-1)
            x = net(x)
        rot_raw = self.rotation_output(x)
        rot = rot_raw / torch.linalg.vector_norm(rot_raw, keepdim=True, dim=-1)
        trans = self.translation_output(x)
        return trans, rot
