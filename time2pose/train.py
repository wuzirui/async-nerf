import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import pypose as pp

import network
from dataloaders import dataloader
from optimization.adaptive_loss import AdaptiveLoss
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
        torch.set_default_dtype(torch.float64)
    
    def setup_dataset(self):
        self.train_dataset = dataloader.load_dataset(self.hparams, split='train')
        self.val_dataset = dataloader.load_dataset(self.hparams, split='val')
        self.test_dataset = None

        if self.hparams.test_datapath is not None:
            self.test_dataset = dataloader.load_dataset(self.hparams, split='test')
    
    def setup_network(self):
        self.network = network.TimePoseFunction(self.hparams).to(self.device)
        self.adaptive_loss_fn = AdaptiveLoss(self.hparams).to(self.device)
        params = []
        params.append({"params": self.network.parameters()})
        params.append({"params": self.adaptive_loss_fn.parameters()})
        self.optimizer = torch.optim.Adam(params=params, lr=self.hparams.lr)
    
    def run(self):
        self.logger.info(self.hparams)
        bar = tqdm(range(self.hparams.train_epochs))
        for epoch in bar:
            result = self._training_step()
            self.write_tensorboard(result, 'train', epoch)
            bar.update(1)

            key_alias = {'loss': 'loss', 'rotation loss': 'l_rot', 'translation loss': 'l_trans', \
                'error_translation_mean': "val/err_trans", 'theta_mean': 'val/theta'}
            bar.set_postfix({key_alias[key]: value for key, value in result.items() if key in key_alias.keys()})
            if epoch % self.hparams.val_interval == 0:
                metrics = self._run_validation()
                self.write_tensorboard(metrics, 'val', epoch)
                metrics.update(result)
                self.logger.info({key_alias[key]: value for key, value in metrics.items() if key in key_alias.keys()})
    
    def write_tensorboard(self, metrics, split, epoch):
        for key, value in metrics.items():
            self.writer.add_scalar(f'{split}/{key}', value, global_step=epoch)

    def _training_step(self):
        cum_metrics = {}
        for batch in self.train_dataset:
            timestamps = batch['timestamp'].reshape(-1, 1).to(self.device)

            #inference through network
            predicted_trans, predicted_rot = self.network(timestamps)

            gt_se3 = batch['SE3'].reshape(-1, 7).to(self.device)
            trans_loss, rot_loss = self.adaptive_loss_fn(torch.cat([predicted_trans, predicted_rot], dim=-1), gt_se3)

            loss = trans_loss + rot_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            metrics = {
                'loss': float(loss),
                'translation loss': float(trans_loss),
                'rotation loss': float(rot_loss),
                'sigma_x': float(self.adaptive_loss_fn.s_x),
                'sigma_q': float(self.adaptive_loss_fn.s_q)
            }

            for key, value in metrics.items():
                if key not in cum_metrics:
                    cum_metrics[key] = value
                else:
                    cum_metrics[key] += value
            
        return cum_metrics    

    def _run_validation(self):
        with torch.inference_mode():
            pred_trans, pred_rot = [], []
            gt_SE3 = []
            for batch in self.val_dataset:
                timestamp = batch['timestamp'].to(self.device)
                gt_SE3.append(batch['SE3'])
                predicted_trans, predicted_rot = self.network(timestamp)
                pred_trans.append(predicted_trans)
                pred_rot.append(predicted_rot)
            pred_trans = torch.cat(pred_trans, dim=0).to(self.device)
            pred_rot = torch.cat(pred_rot, dim=0).to(self.device)
            gt_SE3 = torch.cat(gt_SE3, dim=0).to(self.device)
            error_trans_axes = (pred_trans - gt_SE3[:, :3]).abs().cpu().numpy() * self.hparams.pose_scale_factor
            error_trans = np.linalg.norm(error_trans_axes, axis=-1)
            theta_rot = (torch.acos(torch.sum(pred_rot * gt_SE3[:, 3:], dim=-1).abs().clamp(-1, 1)) * 360 / math.pi).cpu().numpy()
            error_trans_axes_median = np.median(error_trans_axes, axis=1)
            error_trans_axes_mean = np.mean(error_trans_axes, axis=1)
            error_trans_median = np.median(error_trans, axis=0)
            error_trans_mean = np.mean(error_trans, axis=0)
            theta_rot_median = np.median(theta_rot, axis=0)
            theta_rot_mean = np.mean(theta_rot, axis=0)
            metrics = {
                'error_x_median': error_trans_axes_median[0],
                'error_y_median': error_trans_axes_median[1],
                'error_z_median': error_trans_axes_median[2],
                'error_x_mean': error_trans_axes_mean[0],
                'error_y_mean': error_trans_axes_mean[1],
                'error_z_mean': error_trans_axes_mean[2],
                'error_translation_median': error_trans_median,
                'error_translation_mean': error_trans_mean,
                'theta_median': theta_rot_median,
                'theta_mean': theta_rot_mean,
            }
            return metrics
        

if __name__ == '__main__':
    hparams = opts.get_opts_base().parse_args()
    runner = Runner(hparams=hparams)
    runner.run()
