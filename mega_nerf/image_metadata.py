from configparser import Interpolation
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import numpy as np
import torch
import cv2
import torch.nn.functional as F
from PIL import Image
import re


class ImageMetadata:
    def __init__(self, image_path: Path, c2w: torch.Tensor, W: int, H: int, intrinsics: torch.Tensor, image_index: int,
                 mask_path: Optional[Path], is_val: bool, pose_scale_factor, is_depth: bool, gt_pose_path: Optional[Path]):
        self.image_path = image_path
        self.c2w = c2w
        self.W = W
        self.H = H
        self.intrinsics = intrinsics
        self.image_index = image_index
        self._mask_path = mask_path
        self.is_val = is_val
        self.pose_scale_factor = pose_scale_factor
        if gt_pose_path is not None:
            self.gt_pose = torch.tensor(np.loadtxt(gt_pose_path).reshape(3, 4))
        else:
            self.gt_pose = c2w.clone()
        self.is_depth = is_depth
    
    def get_gt_pose(self):
        return self.gt_pose
    
    def is_depth_frame(self) -> bool:
        return self.is_depth
    
    def is_rgb_frame(self) -> bool:
        return not self.is_depth 

    def load_image(self) -> torch.Tensor:
        if self.is_depth_frame():
            return self._load_depth_image()
        return self._load_image()

    def _load_image(self) -> torch.Tensor:
        """
        从文件系统中读取图片, 并按照 metadata 进行缩放
        Returns:
        -  torch.Tensor: 图片的缩放后的 tensor (self.W, self.H, 3)
        """
        rgbs = Image.open(self.image_path).convert('RGB')
        size = rgbs.size

        if size[0] != self.W or size[1] != self.H:
            rgbs = rgbs.resize((self.W, self.H), Image.LANCZOS)

        return torch.ByteTensor(np.asarray(rgbs))
    
    def _load_depth_image(self) -> torch.Tensor:
        """
        read depth image from filesystem
        Returns:
        - torch.Tensor: tensor (self.W, self.H, 1)
        """
        depths = Image.open(self.image_path).convert("L")
        depths = np.asarray(depths, dtype=np.float32)
        depths[depths > 300] = 300
        depths /= self.pose_scale_factor
        depths = np.ascontiguousarray(depths)
        if depths.shape[1] != self.W or depths.shape[0] != self.H:
            depths = cv2.resize(depths, (self.W, self.H), interpolation=cv2.INTER_LANCZOS4)
        depths[depths < 0] = 0
        
        return torch.tensor(depths, dtype=torch.float32)

    def load_mask(self) -> Optional[torch.Tensor]:
        """
        load masks
        Returns:
        - (Optional) torch.Tensor: mask tensor (self.W, self.H)
        """
        if self._mask_path is None:
            return None
        assert (self.is_depth and 'depth' in self._mask_path.parent.parent.stem) or (not self.is_depth and 'rgb' in self._mask_path.parent.parent.stem),\
            f'depth type and metadata not match, is_depth={self.is_depth} and metadata folder path={self._mask_path.parent.parent.stem}'

        with ZipFile(self._mask_path) as zf:
            with zf.open(self._mask_path.name) as f:
                keep_mask = torch.load(f, map_location='cpu')

        if keep_mask.shape[0] != self.H or keep_mask.shape[1] != self.W:
            keep_mask = F.interpolate(keep_mask.unsqueeze(0).unsqueeze(0).float(),
                                      size=(self.H, self.W)).bool().squeeze()

        return keep_mask
