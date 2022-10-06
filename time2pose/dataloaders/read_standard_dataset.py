import numpy as np
import torch.utils.data.dataset as dataset
from pathlib import Path

def read_data(filename):
    with open(filename) as f:
        lines = f.readlines()
    pose = [([eval(num) for num in line.split()]) for line in lines]
    pose = np.array(pose, dtype=np.float32).reshape((-1, 12))
    return pose


class StandardDataset(dataset):
    def __init__(self, datapath, split='train'):
        self.split = split
        assert split in ['train', 'val', 'test']
        self.datapath = Path(datapath)
        if split != 'test':
            self.datapath = self.datapath / 'test.txt'
            assert self.datapath.exists(), "specific dataset split not exists: " + self.datapath.absolute()
            self.frames = np.loadtxt(self.datapath, dtype=np.float32)
            self.n_frames = len(self.frames)
        else:
            self.datapath = self.datapath / split
            assert self.datapath.exists(), "specific dataset split not exists: " + self.datapath.absolute()
            self.frames = [float(x.stem) for x in self.datapath.iterdir() if x.suffix == '.txt']
            self.n_frames = len(self.frames)
            self.poses = [np.loadtxt(x).reshape(4, 4) for x in self.datapath.iterdir() if x.suffix == '.txt']
            

    def __len__(self):
        return self.n_frames
    
    def __getitem__(self):
        pass