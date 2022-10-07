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
                            , format="[%(levelname)s] %(asctime)-9s - %(filename)-8s:%(lineno)s line - %(message)s"
                            , datefmt="%Y-%m-%d %H:%M:%S")
        self.logger = logging.getLogger()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def setup_misc(self):
        torch.manual_seed(self.hparams.random_seed)
        torch.cuda.manual_seed(self.hparams.random_seed)
        torch.cuda.manual_seed_all(self.hparams.random_seed)
    
    def setup_dataset(self):
        self.train_dataset = dataloader.load_dataset(self.hparams, split='train')
        self.val_dataset = dataloader.load_dataset(self.hparams, split='val')
        self.test_dataset = None

        if self.hparams.test_datapath is not None:
            self.test_dataset = dataloader.load_dataset(self.hparams, split='test')
    
    def setup_network(self):
        self.network = network.TimePoseFunction(self.hparams).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.network.parameters(), lr=self.hparams.lr)
    
    def run(self):
        bar = tqdm(range(self.hparams.train_epochs))
        for epoch in bar:
            result = self._training_step()
            self.writer.add_scalar('train/loss', result['loss'], global_step=epoch)
            bar.update(1)
            bar.set_postfix(result)
            if epoch % self.hparams.val_interval == 0:
                self._run_validation()

    def _training_step(self):
        cum = 0
        for batch in self.train_dataset:
            timestamps = batch['timestamp'].reshape(-1, 1).float().to(self.device)
            gt_se3 = batch['SE3'].reshape(-1, 7).float().to(self.device)
            predicted_se3 = self.network(timestamps)
            loss = nn.functional.l1_loss(predicted_se3, gt_se3, reduce='mean')
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            cum += loss.cpu().detach()
        return {
            "loss": cum,
        }
    
    def _run_validation(self):
        pass


if __name__ == '__main__':
    hparams = opts.get_opts_base().parse_args()
    runner = Runner(hparams=hparams)
    runner.run()

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