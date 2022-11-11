import torch
from torch import nn
import math

n_feature_0, n_feature_1 = 100, 50
def hashcode(num: torch.LongTensor) -> torch.LongTensor:
    # return num + 2
    # return ((num ** 13).abs() + 1) % n_feature
    return (num.long() ^ 2654435761 + 1) % n_feature_0, (num.long() ^ 1) % n_feature_1

class TimePoseFunction(nn.Module):
    def __init__(self, hparams):
        super(TimePoseFunction, self).__init__()
        nets = []
        self.use_velocity = hparams.velocity_weight > 0
        n_channels = hparams.n_channels
        self.skip = hparams.skip_connections
        feature_dim = 40
        self.input_dir = feature_dim + 1
        self.feature_dict_0 = nn.Parameter(torch.randn(n_feature_0, feature_dim // 2))
        self.feature_dict_1 = nn.Parameter(torch.randn(n_feature_1, feature_dim // 2))
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
                # nn.Softplus()
                nn.LeakyReLU()
            )
            nets.append(layer)
        
        self.nets= nn.ModuleList(nets)
        self.rotation_output = nn.Sequential(nn.Linear(n_channels, 4))
        self.translation_output = nn.Sequential(nn.Linear(n_channels, 3))

    def _interpolate(self, t):
        t_mid = t.floor().long()
        id_prev_0, id_prev_1 = hashcode(t_mid - 1)
        id_mid_0, id_mid_1 = hashcode(t_mid)
        id_next_0, id_next_1 = hashcode(t_mid + 1)
        feat_prev_0, feat_prev_1 = self.feature_dict_0[id_prev_0].squeeze(1), self.feature_dict_1[id_prev_1].squeeze(1)
        feat_mid_0, feat_mid_1 = self.feature_dict_0[id_mid_0].squeeze(1), self.feature_dict_1[id_mid_1].squeeze(1)
        feat_next_0, feat_next_1 = self.feature_dict_0[id_next_0].squeeze(1), self.feature_dict_1[id_next_1].squeeze(1)
        l_prev = (t - t_mid) * (t - t_mid - 1) / 2
        l_mid = -(t - t_mid + 1) * (t - t_mid -1)
        l_next = (t - t_mid + 1) * (t - t_mid) / 2
        ret = torch.cat([l_prev * feat_prev_0 + l_mid * feat_mid_0 + l_next * feat_next_0, l_prev * feat_prev_1 + l_mid * feat_mid_1 + l_next * feat_next_1], dim=-1)
        return ret
        
    def forward(self, t, train=False):
        if train and self.use_velocity:
            t = t.requires_grad_(True)
        feat = self._interpolate(t)
        x = torch.cat([t, feat], -1)
        for i, net in enumerate(self.nets):
            if i in self.skip:
                x = torch.concat([t, feat, x], dim=-1)
            x = net(x)
        rot_raw = self.rotation_output(x)
        rot = rot_raw / torch.linalg.vector_norm(rot_raw, keepdim=True, dim=-1)
        trans = self.translation_output(x)
        se3 = torch.cat([trans, rot], dim=-1)
        if train and self.use_velocity:
            v  = torch.autograd.grad(se3, t,
                                     grad_outputs=torch.ones([len(t), 7]).to(t.device),
                                     retain_graph=True,
                                     create_graph=True)[0]
        else: v = None
        return trans, rot, v
