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
    pose = np.array(pose, dtype=np.float32).reshape((-1, 12))
    return pose


class DefaultDataset():
    def __init__(self, hparams, split='train'):
        self.logger = logging.getLogger()
        self.split = split
        assert split in ['train', 'val', 'test']
        if split == 'test':
            self.datapath = Path(hparams.test_datapath)
            assert self.datapath.exists(), "specific dataset split not exists: " + str(self.datapath.absolute())
            self.frames = [float(x.stem) for x in self.datapath.iterdir() if x.suffix == '.jpg']
            self.n_frames = len(self.frames)
        else:
            self.datapath = Path(hparams.datapath)
            self.datapath = self.datapath / 'rgb'
            assert self.datapath.exists(), "specific dataset split not exists: " + str(self.datapath.absolute())
            # split datasets
            self.src_files = [x for x in self.datapath.iterdir() if x.suffix == '.txt']
            self.train_idx = [x for i, x in enumerate(self.src_files) if i % (len(self.src_files) // hparams.n_val) != 0]
            self.val_idx = [x for x in self.src_files if x not in self.train_idx]
            self.src_files = [x for x in self.src_files if x in (self.train_idx if split =='train' else self.val_idx)]
            self.n_frames = len(self.src_files)
            self.frames = [float(x.stem) for x in self.src_files if x.suffix == '.txt']
            self.poses_matrices = [np.loadtxt(x).reshape(4, 4) for x in self.datapath.iterdir() if x.suffix == '.txt']
            self.poses_SE3 = [pp.mat2SE3(torch.tensor(x)) for x in self.poses_matrices]
        self.logger.info(f'loading {self.n_frames} frames into datasets, split={split}')
            

    def __len__(self):
        return self.n_frames
    
    def __getitem__(self, i):
        return {
            "timestamp": self.frames[i],
            "SE3": self.poses_SE3[i] if self.split != 'test' else None
        }