import torch
from torch import nn
import pypose as pp

class PoseCorrection(nn.Module):
    
    def __init__(self, n_frames):
        super(PoseCorrection,self).__init__()
        self.correction_dict = nn.Parameter(pp.identity_se3(n_frames))
    
    def forward(self, image_indices, rays, depth_mask):
        correction = torch.where(depth_mask == 1, self.correction_dict[image_indices.long()], pp.identity_se3(len(rays)).to(rays.device))
        correction = pp.Exp(pp.se3(correction))
        ret = torch.zeros_like(rays)
        ret[6:] = rays[6:]
        ret[:, :3] = rays[:, :3] + correction.translation()
        ret[:, 3:6] = (correction.rotation().matrix() @ rays[:, 3:6, None]).squeeze()
        # rays[:, 3:6] = (rays[:,None, 3:6] @ correction.rotation().matrix().transpose(1, 2)).reshape(-1, 3)
        return ret
    
    def forward_c2ws(self, image_indices, c2ws, depth_mask):
        correction = torch.where(depth_mask == 1, self.correction_dict[image_indices.long()], pp.identity_se3(len(c2ws)).to(c2ws.device))
        correction = pp.Exp(pp.se3(correction))
        c2ws_pp = pp.mat2SE3(c2ws.double()).float()     # weird problem of pypose
        ret = correction * c2ws_pp
        return ret
    
    def forward_c2w(self, image_index, c2w):
        correction = pp.Exp(pp.se3(self.correction_dict[image_index]))
        return correction * pp.mat2SE3(c2w.double()).float()
