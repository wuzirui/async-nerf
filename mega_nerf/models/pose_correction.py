import torch
from torch import nn
import pypose as pp

# class PoseCorrection(nn.Module):
    
#     def __init__(self, n_frames):
#         super(PoseCorrection,self).__init__()
#         self.rot_dict = nn.Parameter(pp.identity_so3(n_frames).tensor())
    
#     def forward(self, image_indices, rays, depth_mask):
#         correction = torch.where(depth_mask == 1, self.correction_dict[image_indices.long()], pp.identity_so3(len(rays)).to(rays.device).tensor())
#         correction = pp.Exp(pp.so3(correction))
#         ret = torch.zeros_like(rays)
#         ret[6:] = rays[6:]
#         ret[:, :3] = rays[:, :3]# + correction.translation()
#         ret[:, 3:6] = (correction.matrix() @ rays[:, 3:6, None]).squeeze()
#         # ret[:, 3:6] = (rays[:, None, 3:6] @ correction.matrix().transpose(1,2)).squeeze()
#         # rays[:, 3:6] = (rays[:,None, 3:6] @ correction.rotation().matrix().transpose(1, 2)).reshape(-1, 3)
#         return ret
    
#     def forward_c2ws(self, image_indices, c2ws, depth_mask):
#         correction = torch.where(depth_mask == 1, self.correction_dict[image_indices.long()], pp.identity_so3(len(c2ws)).to(c2ws.device))
#         correction = pp.Exp(pp.so3(correction))
#         c2ws_pp = pp.mat2SE3(c2ws.double()).float()     # weird problem of pypose
#         ret_rot = correction * c2ws_pp.rotation()
#         ret_trans = c2ws_pp.translation()
#         ret = pp.SE3(torch.cat([ret_trans, ret_rot.tensor()], dim=-1))
#         return ret
    
#     def forward_c2w(self, image_index, c2w):
#         correction = pp.Exp(pp.so3(self.correction_dict[image_index]))
#         c2w_pp = pp.mat2SE3(c2w.double()).float()     # weird problem of pypose
#         ret_rot = correction * c2w_pp.rotation()
#         ret_trans = c2w_pp.translation()
#         ret = pp.SE3(torch.cat([ret_trans, ret_rot.tensor()], dim=-1))
#         return ret

# class PoseCorrection(nn.Module):
    
#     def __init__(self, n_frames):
#         super(PoseCorrection,self).__init__()
#         self.correction_dict = nn.Parameter(torch.zeros([n_frames, 3]))
    
#     def forward(self, image_indices, rays, depth_mask):
#         correction = torch.where(depth_mask == 1, self.correction_dict[image_indices.long()], torch.zeros([len(image_indices), 3]).to(rays.device))
#         ret = torch.zeros_like(rays)
#         ret[6:] = rays[6:]
#         ret[:, :3] = rays[:, :3] + correction
#         ret[:, 3:6] = rays[:, 3:6]
#         # rays[:, 3:6] = (rays[:,None, 3:6] @ correction.rotation().matrix().transpose(1, 2)).reshape(-1, 3)
#         return ret
    
#     @torch.no_grad()
#     def forward_c2ws(self, image_indices, c2ws, depth_mask):
#         correction = torch.where(depth_mask == 1, self.correction_dict[image_indices.long()], torch.zeros([len(image_indices), 3]).to(c2ws.device))
#         c2ws[:, :3, 3] += correction
#         ret = pp.mat2SE3(c2ws.double()).float()
#         return ret
    
#     @torch.no_grad()
#     def forward_c2w(self, image_index, c2w):
#         correction = self.correction_dict[image_index]
#         c2w[:3, 3] += correction
#         ret = pp.mat2SE3(c2w.double()).float()
#         return ret

class PoseCorrection(nn.Module):
    
    def __init__(self, n_frames):
        super(PoseCorrection,self).__init__()
        self.rot_dict = nn.Parameter(pp.identity_so3(n_frames).tensor())
        self.trans_dict = nn.Parameter(torch.zeros([n_frames, 3]))
    
    def forward(self, image_indices, rays, depth_mask):
        rot = torch.where(depth_mask == 1, self.rot_dict[image_indices.long()], pp.identity_so3(len(rays)).to(rays.device).tensor())
        trans = torch.where(depth_mask == 1, self.trans_dict[image_indices.long()], torch.zeros([len(rays), 3]).to(rays.device))
        rot = pp.Exp(pp.so3(rot))
        ret = torch.zeros_like(rays)
        ret[6:] = rays[6:]
        ret[:, :3] = rays[:, :3] + trans
        ret[:, 3:6] = (rot.matrix() @ rays[:, 3:6, None]).squeeze()
        return ret
    
    def forward_c2ws(self, image_indices, c2ws, depth_mask):
        correction = torch.where(depth_mask == 1, self.rot_dict[image_indices.long()], pp.identity_so3(len(c2ws)).to(c2ws.device))
        trans = torch.where(depth_mask == 1, self.trans_dict[image_indices.long()], torch.zeros([len(c2ws), 3]).to(c2ws.device))
        correction = pp.Exp(pp.so3(correction))
        c2ws_pp = pp.mat2SE3(c2ws.double()).float()     # weird problem of pypose
        ret_rot = correction * c2ws_pp.rotation()
        ret_trans = c2ws_pp.translation() + trans
        ret = pp.SE3(torch.cat([ret_trans, ret_rot.tensor()], dim=-1))
        return ret
    
    def forward_c2w(self, image_index, c2w):
        rot = pp.Exp(pp.so3(self.rot_dict[image_index]))
        trans = self.trans_dict[image_index]
        c2w_pp = pp.mat2SE3(c2w.double()).float()     # weird problem of pypose
        ret_rot = rot * c2w_pp.rotation()
        ret_trans = c2w_pp.translation() + trans
        ret = pp.SE3(torch.cat([ret_trans, ret_rot.tensor()], dim=-1))
        return ret