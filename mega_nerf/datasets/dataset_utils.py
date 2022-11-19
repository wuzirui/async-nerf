from typing import Tuple, Optional

import torch

from mega_nerf.image_metadata import ImageMetadata


def get_rgbd_index_mask(metadata: ImageMetadata) -> Optional[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    """
    read rgb, depth, index, mask
    Inputs:
    - metadata: image metadata (ImageMetadata)
    Returns:
    - rgbs: (torch.Tensor) shape: (n_mask, 3)
    - depths: (torch.Tensor) shape: (n_mask, 1)
    - indices: (torch.Tensor) shape: (n_mask, 1)
    - masks: pixel-wised mask for training (torch.Tensor) shape: (self.W * self.H, 1)
    """
    if metadata.is_depth:
        image = metadata.load_image().view(-1, 1)
    else:
        image = metadata.load_image().view(-1, 3)

    keep_mask = metadata.load_mask()

    if metadata.is_val:
        if keep_mask is None:
            keep_mask = torch.ones(metadata.H, metadata.W, dtype=torch.bool)
        else:
            # Get how many pixels we're discarding that would otherwise be added
            discard_half = keep_mask[:, metadata.W // 2:]  # mask 的右半边
            discard_pos_count = discard_half[discard_half == True].shape[0]  # 右半边的数据集中的像素数

            candidates_to_add = torch.arange(metadata.H * metadata.W).view(metadata.H, metadata.W)[:, :metadata.W // 2]  # 左半边元素的索引
            keep_half = keep_mask[:, :metadata.W // 2]  # mask 的左半边
            candidates_to_add = candidates_to_add[keep_half == False].reshape(-1)  # 左半边被 mask 掉的元素的索引
            to_add = candidates_to_add[torch.randperm(candidates_to_add.shape[0])[:discard_pos_count]]  # 取其中和右半边 mask 数量相同的元素

            keep_mask.view(-1).scatter_(0, to_add, torch.ones_like(to_add, dtype=torch.bool))  # 重新构成 mask

        keep_mask[:, metadata.W // 2:] = False

    if keep_mask is not None:
        if keep_mask[keep_mask == True].shape[0] == 0:      # return none if no pixel is governed by this centroid
            return None

        keep_mask = keep_mask.view(-1)
        image = image[keep_mask == True]

    assert metadata.image_index <= torch.iinfo(torch.int32).max
    return image, metadata.image_index * torch.ones(image.shape[0], dtype=torch.int32), keep_mask
