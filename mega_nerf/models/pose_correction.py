import torch
from torch import nn
import pypose as pp

class PoseCorrection(nn.Module):
    
    def __init__(self, n_frames):
        super(PoseCorrection,self).__init__()
        self.rotation_corr = nn.Parameter(pp.identity_so3(n_frames))
        self.translation_corr = nn.Parameter(torch.zeros((n_frames, 3, )))
    
    def forward(self, image_indices, rays, depth_mask):
        rotation = torch.where(depth_mask == 1, self.rotation_corr[image_indices.long()], pp.identity_so3(len(rays)).to(rays.device))
        translation = torch.where(depth_mask == 1, self.translation_corr[image_indices.long()], torch.zeros((len(rays), 3, )).to(rays.device))
        rotation = pp.Exp(pp.so3(rotation))
        ret = rays.clone()
        ret[:, :3] = translation + rays[:, :3]
        ret[:, 3:6] = (rotation.matrix() @ rays[:, 3:6, None]).squeeze()
        return ret
    
    def forward_c2ws(self, image_indices, c2ws, depth_mask):
        rotation = torch.where(depth_mask == 1, self.rotation_corr[image_indices.long()], pp.identity_so3(len(c2ws)).to(c2ws.device))
        translation = torch.where(depth_mask == 1, self.translation_corr[image_indices.long()], torch.zeros((len(c2ws), 3, )).to(c2ws.device))
        rotation = pp.Exp(pp.so3(rotation))
        ret = torch.zeros_like(c2ws).to(c2ws.device)
        ret[:, :3, :3] = rotation.matrix() @ c2ws[:, :3, :3]
        ret[:, :3, 3] = translation + c2ws[:, :3, 3]
        return pp.mat2SE3(ret.double()).float()
    
    def forward_c2w(self, image_index, c2w):
        rotation = pp.Exp(pp.so3(self.rotation_corr[image_index]))
        translation = self.translation_corr[image_index]
        ret = torch.eye(4).to(c2w.device)
        ret[:3, :3] = rotation.matrix() @ c2w[:3, :3]
        ret[:3, 3] = translation + c2w[:3, 3]
        return pp.mat2SE3(ret.double()).float()
