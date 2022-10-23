import numpy as np
import torch.utils.data.dataset as dataset
from pathlib import Path
import pypose as pp
import torch
import logging

def read_data(filename):
    with open(filename) as f:
        lines = f.readlines()
    pose = [([eval(num) for num in line.split()]) for line in lines]
    pose = np.array(pose, dtype=np.float64).reshape((-1, 12))
    return pose


class DefaultDataset():
    def __init__(self, hparams, split='train'):
        self.hparams = hparams
        self.logger = logging.getLogger()
        self.split = split
        assert split in ['train', 'val', 'test']
        if split == 'test':
            self.datapath = Path(hparams.test_datapath)
            assert self.datapath.exists(), "specific dataset split not exists: " + str(self.datapath.absolute())
            self.frames = [float(x.stem) - hparams.start_timestamp for x in self.datapath.iterdir() if x.suffix == '.jpg']
            self.n_frames = len(self.frames)
        else:
            self.datapath = Path(hparams.datapath)
            self.datapath = self.datapath
            assert self.datapath.exists(), "specific dataset split not exists: " + str(self.datapath.absolute())
            # split datasets
            self.src_files = [x for x in self.datapath.iterdir() if x.suffix == '.txt']
            self.src_files = sorted(self.src_files, key=lambda x: float(x.stem))

            timestamps = [float(x.stem) - hparams.start_timestamp for x in self.src_files]
            print(f'min timestamp = {min(timestamps)}, max timestamp = {max(timestamps)}')
            
            self.train_idx = [x for i, x in enumerate(self.src_files) if i % (len(self.src_files) // hparams.n_val) != 0]
            self.val_idx = [x for x in self.src_files if x not in self.train_idx]
            self.src_files = sorted([x for x in self.src_files if x in (self.train_idx if split =='train' else self.val_idx)], key=lambda x: float(x.stem))
            self.n_frames = len(self.src_files)
            self.frames = [[float(x.stem) - hparams.start_timestamp] for x in self.src_files if x.suffix == '.txt']
            self.frames = torch.tensor(self.frames)
            self.poses_matrices = [torch.tensor(np.loadtxt(x), dtype=torch.float64).reshape(4, 4) for x in self.src_files if x.suffix == '.txt']
            self.poses_matrices = torch.cat(self.poses_matrices, dim=0).reshape(-1, 4, 4)
            self.logger.info(f'mean translation = {torch.mean(self.poses_matrices[:, :3, 3], dim=0)}')
            self.poses_matrices[:, :3, 3] -= torch.tensor(hparams.centroid)
            self.poses_matrices[:, :3, 3] /= hparams.pose_scale_factor
            self.poses_SE3 = pp.mat2SE3(self.poses_matrices)
            if self.split == 'train':
                self._calc_velocity()
        self.logger.info(f'loaded {self.n_frames} frames into datasets, split={split}')

    def __len__(self):
        return self.n_frames
    
    def __getitem__(self, i):
        ret = {
            "timestamp": self.frames[i],
            "SE3": self.poses_SE3[i] if self.split != 'test' else torch.tensor(0),
        }
        if self.split == 'train':
            ret['velocity'] = self.v[i]
        return ret
    
    def _calc_velocity(self):
        dx = self.poses_SE3[1:].tensor() - self.poses_SE3[:-1].tensor()
        dt = self.frames[1:] - self.frames[:-1]
        v = dx / dt
        self.v = torch.cat([v, v[-1:]], dim=0)