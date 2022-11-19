import torch
from torch import nn
import math



class TimePoseFunction(nn.Module):
    def __init__(self, hparams):
        super(TimePoseFunction, self).__init__()
        nets = []
        self.use_velocity = hparams.velocity_weight > 0
        self.hparams = hparams
        n_channels = hparams.n_channels
        self.skip = hparams.skip_connections
        feature_dim = 2
        if hparams.feature_type in ['grid', 'hash']:
            self.feature_dict = [nn.Parameter(torch.randn(n_feat, hparams.feature_dim)) for n_feat in hparams.n_grid_features]
            feature_dim = len(hparams.n_grid_features * hparams.feature_dim)
            
        self.input_dir = feature_dim + 1
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

    def hashcode(self, num: torch.LongTensor) -> torch.LongTensor:
        if self.hparams.feature_type is None:
            return None
        elif self.hparams.feature_type == 'grid':
            assert num.max() + 1 < self.hparams.n_grid_features[0], "index out of range"
            return num + 1
        elif self.hparams.feature_type == 'hash':
            ret = [((num ^ x) * 10) % mod for x, mod in zip(self.hparams.hash_parameters, self.hparams.n_grid_features)]
            return ret
        else:
            raise "feature type not found"

    def _interpolate(self, t):
        t_mid = t.floor().long()
        id_prev = self.hashcode(t_mid - 1)
        id_mid = self.hashcode(t_mid)
        id_next = self.hashcode(t_mid + 1)
        feat_prev = [feat_dict[idx].squeeze(1).to(t.device) for feat_dict, idx in zip(self.feature_dict, id_prev)]
        feat_mid = [feat_dict[idx].squeeze(1).to(t.device) for feat_dict, idx in zip(self.feature_dict, id_mid)]
        feat_next = [feat_dict[idx].squeeze(1).to(t.device) for feat_dict, idx in zip(self.feature_dict, id_next)]
        l_prev = (t - t_mid) * (t - t_mid - 1) / 2
        l_mid = -(t - t_mid + 1) * (t - t_mid -1)
        l_next = (t - t_mid + 1) * (t - t_mid) / 2
        ret = torch.cat([l_prev * previ + l_mid * midi + l_next * nexti for previ, midi, nexti in zip(feat_prev, feat_mid, feat_next)], dim=-1)
        return ret
        
    def forward(self, t, train=False):
        if train and self.use_velocity:
            t = t.requires_grad_(True)
        if self.hparams.feature_type is None:
            feat = torch.cat([torch.zeros_like(t), torch.ones_like(t)], dim=-1)
        else:
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
