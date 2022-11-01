import torch
from torch import nn
import pypose as pp

def vec2ss_matrix(vector):  # vector to skewsym. matrix
    if vector.dim() == 1:
        ss_matrix = torch.zeros((3,3))
        ss_matrix[0, 1] = -vector[2]
        ss_matrix[0, 2] = vector[1]
        ss_matrix[1, 0] = vector[2]
        ss_matrix[1, 2] = -vector[0]
        ss_matrix[2, 0] = -vector[1]
        ss_matrix[2, 1] = vector[0]

        return ss_matrix
    else:
        assert vector.dim() == 2
        ss_matrix = torch.zeros((len(vector), 3,3))
        ss_matrix[:, 0, 1] = -vector[:, 2]
        ss_matrix[:, 0, 2] = vector[:, 1]
        ss_matrix[:, 1, 0] = vector[:, 2]
        ss_matrix[:, 1, 2] = -vector[:, 0]
        ss_matrix[:, 2, 0] = -vector[:, 1]
        ss_matrix[:, 2, 1] = vector[:, 0]

        return ss_matrix

class PoseCorrection(nn.Module):
    
    def __init__(self, n_frames):
        super(PoseCorrection,self).__init__()
        self.w = nn.Parameter(torch.normal(0., 1e-6, size=(n_frames, 3, )))
        self.v = nn.Parameter(torch.normal(0., 1e-6, size=(n_frames, 3, )))
        self.theta = nn.Parameter(torch.normal(0., 1e-6, size=(n_frames, )))
        self.n_frames = n_frames
    
    def _get_correction(self, image_indices, depth_mask):
        device = image_indices.device
        corr_mat = torch.eye(4).expand((len(image_indices), 4, 4)).float().to(device)
        depth_indices = image_indices.unsqueeze(1)[depth_mask == 1].long()
        assert depth_indices.max() < self.n_frames and depth_indices.min() >= 0
        w_skewsym = vec2ss_matrix(self.w[depth_indices]).to(device)
        theta = self.theta[depth_indices]
        corr_mat[(depth_mask == 1).squeeze(), :3, :3] = torch.eye(3).expand((len(depth_indices), 3, 3)).to(device) + \
            torch.sin(theta)[:, None, None] * w_skewsym + \
            (1 - torch.cos(theta))[:, None, None] * torch.matmul(w_skewsym, w_skewsym)
        corr_mat[(depth_mask == 1).squeeze(), :3, 3] = torch.matmul(torch.eye(3).expand((len(depth_indices), 3, 3)).to(device) * theta[:, None, None] + \
            (1 - torch.cos(theta))[:, None, None] * w_skewsym + (theta - torch.sin(theta))[:, None, None] * \
                torch.matmul(w_skewsym, w_skewsym), self.v[depth_indices, :, None]).squeeze(-1).float()
        return corr_mat
    
    def forward(self, image_indices, rays, depth_mask):
        corr = self._get_correction(image_indices, depth_mask)
        rays[:, :3] += corr[:, :3, 3]
        rays[:, 3:6] = (corr[:, :3, :3] @ rays[:, 3:6, None]).squeeze()
        return rays
    
    def forward_c2ws(self, image_indices, c2ws, depth_mask):
        corr = self._get_correction(image_indices, depth_mask)
        return (pp.mat2SE3(corr.double()) * pp.mat2SE3(c2ws.double())).float()
    
    def forward_c2w(self, image_index, c2w):
        device = c2w.device
        corr_mat = torch.eye(4).float().to(device)
        w_skewsym = vec2ss_matrix(self.w[image_index]).to(device)
        theta = self.theta[image_index]
        corr_mat[:3, :3] = torch.eye(3).to(device) + torch.sin(theta) * w_skewsym + \
            (1 - torch.cos(theta)) * torch.matmul(w_skewsym, w_skewsym)
        corr_mat[:3, 3] = torch.matmul(torch.eye(3).to(device) * theta + \
            (1 - torch.cos(theta)) * w_skewsym + (theta - torch.sin(theta)) * \
                torch.matmul(w_skewsym, w_skewsym), self.v[image_index, :, None]).squeeze(-1).float()
        return (pp.mat2SE3(corr_mat.double()) * pp.mat2SE3(c2w.double())).float()
