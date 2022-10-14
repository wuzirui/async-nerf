import torch
from torch import nn
import pypose as pp

class PoseCorrection(nn.Module):
    
    def __init__(self, n_frames):
        super(PoseCorrection,self).__init__()
        self.correction_dict = nn.Parameter(pp.identity_SE3(n_frames))
    
    def forward(self, image_indices, rays, depth_mask):
        correction = torch.where(depth_mask == 1, self.correction_dict[image_indices], pp.identity_SE3(len(rays)))
        rays[:, :3] += correction.translation()
        rays[:, 3:] = correction.rotation().matrix() * rays[:, 3:, None]
        return rays
        