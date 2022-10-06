import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import network
from dataloaders import dataloader
import opts


class Runner:
    def __init__(self, hparams):
        self.hparams = hparams
        self.setup_environment()
        self.setup_misc()
        self.setup_dataset()
        self.setup_network()
    
    def setup_environment(self):
        self.exp_folder = Path(hparams.exp_name)
        if not self.exp_folder.exists():
            os.makedirs(self.exp_folder.absolute())
        self.exp_name = str(max([int(name.name) for name in self.exp_folder.iterdir()] + [0]) + 1)
        self.writer = SummaryWriter(self.exp_folder / self.exp_name / 'tb')
        logging.basicConfig(level=logging.DEBUG
                            , format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s"
                            , datefmt="%Y-%m-%d %H:%M:%S")
        self.logger = logging.getLogger()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def setup_misc(self):
        torch.manual_seed(hparams.random_seed)
        torch.cuda.manual_seed(hparams.random_seed)
        torch.cuda.manual_seed_all(hparams.random_seed)
    
    def setup_dataset(self):
        train_pose = dataloader.load_dataset(hparams.train_datapath, hparams.dataset_type)
    
    def setup_network(self):
        self.network = network.TimePoseFunction(self.hparams)
    
    def run():
        pass


if __name__ == '__main__':
    hparams = opts.get_opts_base().parse_args()
    runner = Runner(hparams=hparams)

    # n_frames = len(pose)
    # ts = np.linspace(0, n_frames - 1, n_frames)
    # pose, ts = torch.tensor(pose, dtype=torch.float32).to(device), torch.tensor(ts, dtype=torch.float32).to(device)
    # ts = ts.view(-1, 1)
    # train_idx = [i for i in range(n_frames) if i % valset_interval != 0]
    # val_idx = [i for i in range(n_frames) if i % valset_interval == 0]
    # n_train = len(train_idx)
    # n_val = len(val_idx)

    # logger.info(f"frame num: {n_frames}, train set: {n_train}, val set: {n_val}")
    # train_ts, val_ts = ts[train_idx], ts[val_idx]
    # train_pose, val_pose = pose[train_idx], pose[val_idx]
    # network = TimePoseFunction().to(device)
    # optimizer = torch.optim.Adam(list(network.parameters()), lr=1e-2)
    # train_epoch = 100000
    # val_interval = 2000

    # bar = tqdm(range(train_epoch))
    # for epoch in bar:
    #     output = network(train_ts)
    #     optimizer.zero_grad()
    #     loss = F.mse_loss(output, train_pose, reduction='mean')
    #     loss.backward()
    #     optimizer.step()
    #     writer.add_scalar('train/loss', loss, global_step=epoch)

    #     bar.set_postfix_str(f'loss={loss}')
    #     if epoch % val_interval == 0 and epoch != 0:
    #         with torch.inference_mode(True):
    #             output = network(val_ts)
    #             loss = F.mse_loss(output, val_pose, reduction='mean')
    #             writer.add_scalar('val/loss', loss, global_step=epoch)
    #             print(f'validation loss = {loss}')

    # torch.save(network, "checkpoint.pt")
    # test_ts = ts
    # output = network(test_ts).cpu().detach().numpy()

    # output_filename = 'manmade_12_out.txt'
    # np.savetxt(output_filename, output)