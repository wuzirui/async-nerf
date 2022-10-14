import torch
import torch.nn.functional as F
import torch.nn as nn
import pypose as pp


class AdaptiveLoss(nn.Module):

    def __init__(self, hparams):
        """
        :param config: (dict) configuration to determine behavior
        """
        super(AdaptiveLoss, self).__init__()
        self.learnable = True
        self.s_x = torch.nn.Parameter(torch.Tensor([hparams.translation_weight]), requires_grad=self.learnable)
        self.s_q = torch.nn.Parameter(torch.Tensor([hparams.rotation_weight]), requires_grad=self.learnable)
        self.norm_x = 1 if hparams.trans_loss_type == 'l1' else 2
        self.norm_q = 1 if hparams.rot_loss_type == 'l1' else 2

    def forward(self, est_pose, gt_pose):
            """
            Forward pass
            :param est_pose: (torch.Tensor) batch of estimated poses, a Nx7 tensor
            :param gt_pose: (torch.Tensor) batch of ground_truth poses, a Nx7 tensor
            :return: camera pose loss
            """
            # Position loss
            l_x = torch.norm(gt_pose.translation() - est_pose[:, 0:3], dim=1, p=self.norm_x).mean()
            # Orientation loss (normalized to unit norm)
            l_q = torch.norm(F.normalize(gt_pose.rotation(), p=2, dim=1) - F.normalize(est_pose[:, 3:], p=2, dim=1),
                             dim=1, p=self.norm_q).mean()

            if self.learnable:
                return l_x * torch.exp(-self.s_x) + self.s_x, l_q * torch.exp(-self.s_q) + self.s_q
            else:
                return self.s_x*l_x , self.s_q*l_q