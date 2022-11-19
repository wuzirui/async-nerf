import datetime
import faulthandler
from logging import warning
import math
import os
import random
import shutil
import signal
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import pypose as pp
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from mega_nerf.datasets.filesystem_dataset import FilesystemDataset
from mega_nerf.datasets.memory_dataset import MemoryDataset
from mega_nerf.image_metadata import ImageMetadata
from mega_nerf.metrics import depth_abs_rel, depth_delta, depth_rmse, depth_rmse_log, depth_sq_rel, psnr, ssim, lpips
from mega_nerf.misc_utils import main_print, main_tqdm
from mega_nerf.models.model_utils import get_nerf, get_bg_nerf
from mega_nerf.models.pose_correction import PoseCorrection
from mega_nerf.ray_utils import get_rays, get_ray_directions
from mega_nerf.rendering import render_rays


class Runner:
    def __init__(self, hparams: Namespace, set_experiment_path: bool = True):
        faulthandler.register(signal.SIGUSR1)

        if hparams.ckpt_path is not None:
            # retrieve random generator state
            checkpoint = torch.load(hparams.ckpt_path, map_location='cpu')
            np.random.set_state(checkpoint['np_random_state'])
            torch.set_rng_state(checkpoint['torch_random_state'])
            random.setstate(checkpoint['random_state'])
        else:
            np.random.seed(hparams.random_seed)
            torch.manual_seed(hparams.random_seed)
            random.seed(hparams.random_seed)

        self.hparams = hparams

        if 'RANK' in os.environ:
            dist.init_process_group(backend='nccl', timeout=datetime.timedelta(0, hours=24))
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
            self.is_master = (int(os.environ['RANK']) == 0)
        else:
            self.is_master = True

        self.is_local_master = ('RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0
        main_print(hparams)

        if set_experiment_path:
            self.experiment_path = self._get_experiment_path() if self.is_master else None
            self.model_path = self.experiment_path / 'models' if self.is_master else None

        self.writer = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        """
        load datasets
        - coordinates.pt
            - origin_drb: original point position (down-right-back), torch.Tensor, shape = (3)
            - scale_factor
        """
        coordinate_info = torch.load(Path(hparams.dataset_path) / 'coordinates.pt', map_location='cpu')
        self.origin_drb = coordinate_info['origin_drb']
        self.pose_scale_factor = coordinate_info['pose_scale_factor']
        main_print('Origin: {}, scale factor: {}'.format(self.origin_drb, self.pose_scale_factor))

        self.near = hparams.near / self.pose_scale_factor

        if self.hparams.far is not None:
            self.far = hparams.far / self.pose_scale_factor
        elif hparams.bg_nerf:
            self.far = 1e5
        else:
            self.far = 2

        main_print('Ray bounds: {}, {}'.format(self.near, self.far))

        """
        https://github.com/cmusatyalab/mega-nerf/issues/6#issuecomment-1103187971
        """
        self.ray_altitude_range = [(x - self.origin_drb[0]) / self.pose_scale_factor for x in
                                   hparams.ray_altitude_range] if hparams.ray_altitude_range is not None else None
        main_print('Ray altitude range in [-1, 1] space: {}'.format(self.ray_altitude_range))
        main_print('Ray altitude range in metric space: {}'.format(hparams.ray_altitude_range))

        if self.ray_altitude_range is not None:
            assert self.ray_altitude_range[0] < self.ray_altitude_range[1]

        """
        load masks
        - check the consistency of the meta informations
        """
        if self.hparams.cluster_mask_path is not None:
            color_mask_path = Path(self.hparams.cluster_mask_path) / 'mask_rgb'
            cluster_params = torch.load(color_mask_path / 'params.pt', map_location='cpu')
            assert cluster_params['near'] == self.near
            assert (torch.allclose(cluster_params['origin_drb'], self.origin_drb))
            assert cluster_params['pose_scale_factor'] == self.pose_scale_factor

            if self.ray_altitude_range is not None:
                assert (torch.allclose(torch.FloatTensor(cluster_params['ray_altitude_range']),
                                       torch.FloatTensor(self.ray_altitude_range))), \
                    '{} {}'.format(self.ray_altitude_range, cluster_params['ray_altitude_range'])

        self.train_items, self.val_items = self._get_image_metadata()
        main_print('Using {} train images and {} val images'.format(len(self.train_items), len(self.val_items)))
        main_print(f'{self.n_depth_frames} depth frames in total (train + val).')

        camera_positions = torch.cat([x.c2w[:3, 3].unsqueeze(0) for x in self.train_items + self.val_items])
        min_position = camera_positions.min(dim=0)[0]
        max_position = camera_positions.max(dim=0)[0]

        main_print('Camera range in metric space: {} {}'.format(min_position * self.pose_scale_factor + self.origin_drb,
                                                                max_position * self.pose_scale_factor + self.origin_drb))

        main_print('Camera range in [-1, 1] space: {} {}'.format(min_position, max_position))

        """
        initialize models
        """
        self.nerf = get_nerf(hparams, len(self.train_items)).to(self.device)
        if 'RANK' in os.environ:
            self.nerf = torch.nn.parallel.DistributedDataParallel(self.nerf, device_ids=[int(os.environ['LOCAL_RANK'])],
                                                                  output_device=int(os.environ['LOCAL_RANK']))

        if hparams.BARF:
            self.pose_correction = PoseCorrection(self.n_depth_frames).to(self.device)
            if 'RANK' in os.environ:
                self.pose_correction = torch.nn.parallel.DistributedDataParallel(self.pose_correction, device_ids=[int(os.environ['LOCAL_RANK'])],
                                                                                 output_device=int(os.environ['LOCAL_RANK']))
        else:
            self.pose_correction = None
            
        if hparams.bg_nerf:
            self.bg_nerf = get_bg_nerf(hparams, len(self.train_items)).to(self.device)
            if 'RANK' in os.environ:
                self.bg_nerf = torch.nn.parallel.DistributedDataParallel(self.bg_nerf,
                                                                         device_ids=[int(os.environ['LOCAL_RANK'])],
                                                                         output_device=int(os.environ['LOCAL_RANK']))

            if hparams.ellipse_bounds:
                assert hparams.ray_altitude_range is not None

                if self.ray_altitude_range is not None:
                    ground_poses = camera_positions.clone()
                    ground_poses[:, 0] = self.ray_altitude_range[1]
                    air_poses = camera_positions.clone()
                    air_poses[:, 0] = self.ray_altitude_range[0]
                    used_positions = torch.cat([camera_positions, air_poses, ground_poses])
                else:
                    used_positions = camera_positions

                max_position[0] = self.ray_altitude_range[1]
                main_print('Camera range in [-1, 1] space with ray altitude range: {} {}'.format(min_position,
                                                                                                 max_position))

                self.sphere_center = ((max_position + min_position) * 0.5).to(self.device)
                self.sphere_radius = ((max_position - min_position) * 0.5).to(self.device)
                scale_factor = ((used_positions.to(self.device) - self.sphere_center) / self.sphere_radius).norm(
                    dim=-1).max()

                self.sphere_radius *= (scale_factor * hparams.ellipse_scale_factor)
            else:
                self.sphere_center = None
                self.sphere_radius = None

            main_print('Sphere center: {}, radius: {}'.format(self.sphere_center, self.sphere_radius))
        else:
            self.bg_nerf = None
            self.sphere_center = None
            self.sphere_radius = None
        # end __init__

    def train(self):
        self._setup_experiment_dir()

        scaler = torch.cuda.amp.GradScaler(enabled=self.hparams.amp)

        """
        初始化 optimizers, 使用 Adam 训练, 前后景模型使用两个不同的 optimizer
        """
        optimizers = {}
        optimizers['nerf'] = Adam(self.nerf.parameters(), lr=self.hparams.lr)
        if self.bg_nerf is not None:
            optimizers['bg_nerf'] = Adam(self.bg_nerf.parameters(), lr=self.hparams.lr)

        if self.hparams.BARF:
            optimizers['poses'] = Adam(self.pose_correction.parameters(), lr=self.hparams.lr_pose)

        self.iter_step = -1
        if self.hparams.ckpt_path is not None:
            """
            从 checkpoint 恢复模型参数
            """
            checkpoint = torch.load(self.hparams.ckpt_path, map_location='cpu')
            train_iterations = checkpoint['iteration']
            self.iter_step = train_iterations

            scaler_dict = scaler.state_dict()
            scaler_dict.update(checkpoint['scaler'])
            scaler.load_state_dict(scaler_dict)

            for key, optimizer in optimizers.items():
                optimizer_dict = optimizer.state_dict()
                if key == 'poses':
                    continue
                optimizer_dict.update(checkpoint['optimizers'][key])
                optimizer.load_state_dict(optimizer_dict)
            discard_index = checkpoint['dataset_index'] if self.hparams.resume_ckpt_state else -1
        else:
            train_iterations = 0
            discard_index = -1

        schedulers = {}
        for key, optimizer in optimizers.items():
            if key == 'poses':
                continue
            schedulers[key] = ExponentialLR(optimizer,
                                            gamma=self.hparams.lr_decay_factor ** (1 / self.hparams.train_iterations),
                                            last_epoch=train_iterations - 1)

        if self.hparams.dataset_type == 'filesystem':
            # Let the local master write data to disk first
            # We could further parallelize the disk writing process by having all of the ranks write data,
            # but it would make determinism trickier
            if 'RANK' in os.environ and (not self.is_local_master):
                dist.barrier()

            dataset = FilesystemDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                        self.hparams.center_pixels, self.device,
                                        [Path(x) for x in sorted(self.hparams.chunk_paths)], self.hparams.num_chunks,
                                        self.hparams.train_scale_factor, self.hparams.disk_flush_size)
            if self.hparams.ckpt_path is not None and self.hparams.resume_ckpt_state:
                dataset.set_state(checkpoint['dataset_state'])
            if 'RANK' in os.environ and self.is_local_master:
                dist.barrier()
        elif self.hparams.dataset_type == 'memory':
            dataset = MemoryDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device)
        else:
            raise Exception('Unrecognized dataset type: {}'.format(self.hparams.dataset_type))

        if self.is_master:
            pbar = tqdm(total=self.hparams.train_iterations)
            pbar.update(train_iterations)
        else:
            pbar = None

        torch.set_default_dtype(torch.float32)
        while train_iterations < self.hparams.train_iterations:
            # If discard_index >= 0, we already set to the right chunk through set_state
            if self.hparams.dataset_type == 'filesystem' and discard_index == -1:
                dataset.load_chunk()

            if 'RANK' in os.environ:
                world_size = int(os.environ['WORLD_SIZE'])
                sampler = DistributedSampler(dataset, world_size, int(os.environ['RANK']))
                assert self.hparams.batch_size % world_size == 0
                data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size // world_size, sampler=sampler,
                                         num_workers=0, pin_memory=True)
            else:
                data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0,
                                         pin_memory=True)

            for dataset_index, item in enumerate(data_loader):
                if dataset_index <= discard_index:
                    continue

                discard_index = -1

                with torch.cuda.amp.autocast(enabled=self.hparams.amp):
                    image_indices = item['img_indices'].to(self.device, non_blocking=True)

                    metrics, bg_nerf_rays_present = self._training_step(
                        rgbs=item['rgbs'].to(self.device, non_blocking=True) if item['rgbs'] is not None else None,
                        depths=item['depths'].to(self.device, non_blocking=True) if item['depths'] is not None else None,
                        rays=item['rays'].to(self.device, non_blocking=True),
                        image_indices=image_indices,
                        depth_masks=item['depth_mask'].to(self.device, non_blocking=True),
                        c2ws=item['c2ws'].to(self.device, non_blocking=True), 
                        gt_c2ws=item['gt_c2ws'].to(self.device, non_blocking=True))

                    with torch.no_grad():
                        for key, val in metrics.items():
                            if key == 'psnr' and math.isinf(val):  # a perfect reproduction will give PSNR = infinity
                                continue

                            if not math.isfinite(val):
                                warning('Train metrics not finite: {}'.format(metrics))
                                break

                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)

                if not math.isfinite(metrics['loss']):
                    continue
                scaler.scale(metrics['loss']).backward()

                for key, optimizer in optimizers.items():
                    if key == 'bg_nerf' and (not bg_nerf_rays_present):
                        continue
                    elif key =='poses' and self.progress < self.hparams.BARF_start:
                        continue
                    else:
                        scaler.step(optimizer)

                scaler.update()

                for key, scheduler in schedulers.items():
                    if key == 'bg_nerf' and (not bg_nerf_rays_present):
                        continue
                    elif key =='poses' and self.progress < self.hparams.BARF_start:
                        continue
                    scheduler.step()

                train_iterations += 1
                if self.is_master:
                    pbar.update(1)
                    for key, value in metrics.items():
                        self.writer.add_scalar('train/{}'.format(key), value, train_iterations)

                    if train_iterations > 0 and train_iterations % self.hparams.ckpt_interval == 0:
                        self._save_checkpoint(optimizers, scaler, train_iterations, dataset_index,
                                              dataset.get_state() if self.hparams.dataset_type == 'filesystem' else None)

                if train_iterations % self.hparams.val_interval == 1:
                    self._run_validation(train_iterations)

                if train_iterations >= self.hparams.train_iterations:
                    break

        if 'RANK' in os.environ:
            dist.barrier()

        if self.is_master:
            pbar.close()
            self._save_checkpoint(optimizers, scaler, train_iterations, dataset_index,
                                  dataset.get_state() if self.hparams.dataset_type == 'filesystem' else None)

        if self.hparams.cluster_mask_path is None:
            val_metrics = self._run_validation(train_iterations)
            self._write_final_metrics(val_metrics)

    def eval(self):
        self._setup_experiment_dir()
        val_metrics = self._run_validation(0)
        self._write_final_metrics(val_metrics)

    def _write_final_metrics(self, val_metrics: Dict[str, float]) -> None:
        if self.is_master:
            with (self.experiment_path / 'metrics.txt').open('w') as f:
                for key in val_metrics:
                    avg_val = val_metrics[key] / len(self.val_items)
                    message = 'Average {}: {}'.format(key, avg_val)
                    main_print(message)
                    f.write('{}\n'.format(message))

            self.writer.flush()
            self.writer.close()

    def _setup_experiment_dir(self) -> None:
        if self.is_master:
            self.experiment_path.mkdir()
            with (self.experiment_path / 'hparams.txt').open('w') as f:
                for key in vars(self.hparams):
                    f.write('{}: {}\n'.format(key, vars(self.hparams)[key]))
                if 'WORLD_SIZE' in os.environ:
                    f.write('WORLD_SIZE: {}\n'.format(os.environ['WORLD_SIZE']))

            with (self.experiment_path / 'command.txt').open('w') as f:
                f.write(' '.join(sys.argv))
                f.write('\n')

            self.model_path.mkdir(parents=True)

            with (self.experiment_path / 'image_indices.txt').open('w') as f:
                for i, metadata_item in enumerate(self.train_items):
                    f.write('{},{}\n'.format(metadata_item.image_index, metadata_item.image_path.name))
        self.writer = SummaryWriter(str(self.experiment_path / 'tb')) if self.is_master else None

        if 'RANK' in os.environ:
            dist.barrier()
        # end _setup_experiment_dir

    def _training_step(self, rgbs: torch.Tensor, depths: torch.Tensor, rays: torch.Tensor, image_indices: Optional[torch.Tensor], depth_masks: torch.BoolTensor, c2ws: torch.FloatTensor, gt_c2ws: torch.FloatTensor) \
            -> Tuple[Dict[str, Union[torch.Tensor, float]], bool]:
        self.iter_step += 1
        if self.pose_correction is not None:
            rays = self.pose_correction(image_indices, rays, depth_masks)
            c2ws = self.pose_correction.forward_c2ws(image_indices, c2ws, depth_masks)
            gt_c2ws = pp.mat2SE3(gt_c2ws.double()).float()
        self.progress = self.iter_step/self.hparams.train_iterations
        results, bg_nerf_rays_present = render_rays(nerf=self.nerf,
                                                    bg_nerf=self.bg_nerf,
                                                    rays=rays,
                                                    image_indices=image_indices,
                                                    hparams=self.hparams,
                                                    sphere_center=self.sphere_center,
                                                    sphere_radius=self.sphere_radius,
                                                    get_depth=True,
                                                    get_depth_variance=True,
                                                    get_bg_fg_rgb=False,
                                                    progress=self.progress,
                                                    )
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        # print(f"rays shape = {rays.shape}, rendered rgb shape = {results[f'rgb_{typ}'].shape}, rendered depth shape = {results[f'depth_{typ}'].shape}")

        color_masks = 1 - depth_masks
        metrics = {}
        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'] * color_masks, rgbs * color_masks)
            depth_variance_metrics = results[f'depth_variance_{typ}'] * (self.pose_scale_factor ** 2) + 1e-4
            if self.pose_correction is not None and self.hparams.have_gt_poses:
                error_trans_axes = (c2ws.translation() - gt_c2ws.translation()).abs().cpu().numpy() * self.pose_scale_factor
                mask = (depth_masks.cpu() == 1).reshape(-1)
                error_trans = np.linalg.norm(error_trans_axes, axis=-1)[mask]
                theta_rot = (torch.acos(torch.sum(c2ws.rotation().tensor() * gt_c2ws.rotation().tensor(), dim=-1).abs().clamp(-1, 1)) * 360 / math.pi)[mask].cpu().numpy()
                error_trans_median = np.median(error_trans, axis=0)
                error_trans_mean = np.mean(error_trans, axis=0)
                theta_rot_median = np.median(theta_rot, axis=0)
                theta_rot_mean = np.mean(theta_rot, axis=0)
                metrics.update({
                    'rot_mse_mean': theta_rot_mean,
                    'rot_mse_median': theta_rot_median,
                    'trans_mse_mean': error_trans_mean,
                    'trans_mse_median': error_trans_median,
                })

        metrics.update({
            'psnr': psnr_,
            'depth_variance': depth_variance_metrics.mean(),
        })

        photo_loss = F.mse_loss(results[f'rgb_{typ}'] * color_masks, rgbs * color_masks, reduction='mean') * self.hparams.photo_weight

        depths_metric, render_depth_metric = (depths * self.pose_scale_factor).view(-1), \
            (results[f'depth_{typ}'].reshape(-1, 1) * depth_masks * self.pose_scale_factor).view(-1)
        metrics.update({
            "train/RMSE": depth_rmse(render_depth_metric, depths_metric),
            "train/RMSE_log": depth_rmse_log(render_depth_metric, depths_metric),
            "train/Abs_Rel": depth_abs_rel(render_depth_metric, depths_metric),
            "train/Sq_Rel": depth_sq_rel(render_depth_metric, depths_metric),
            "train/δ_1": depth_delta(render_depth_metric, depths_metric, 1),
            "train/δ_2": depth_delta(render_depth_metric, depths_metric, 2),
            "train/δ_3": depth_delta(render_depth_metric, depths_metric, 3),
        })
        random_mask = torch.where(torch.randn_like(render_depth_metric) > 0.5, torch.ones_like(render_depth_metric), torch.zeros_like(render_depth_metric)).to(render_depth_metric.device) == 1
        depth_loss = F.mse_loss(render_depth_metric[random_mask], depths_metric[random_mask], reduction='mean')
        pose_mask = torch.logical_and(torch.logical_and(depths_metric > 1e-5, depths_metric < 85), render_depth_metric < 85).logical_and(random_mask == 1)
        depth_pose_loss = F.l1_loss((render_depth_metric[pose_mask] + 1e-5).log(), (depths_metric[pose_mask] + 1e-5).log(), reduction='mean')
        if not depth_pose_loss.isfinite(): depth_pose_loss = 0
        metrics['photo_loss'] = photo_loss
        metrics['depth_mse_loss'] = depth_loss
        metrics['depth_pose_loss'] = depth_pose_loss
        depth_weight = self.hparams.depth_weight * self.progress if self.progress >= self.hparams.BARF_start else 0.
        metrics['loss'] = depth_loss * depth_weight + depth_pose_loss * (self.hparams.depth_weight - depth_weight) +  photo_loss
        return metrics, bg_nerf_rays_present

    def _run_validation(self, train_index: int) -> Dict[str, float]:
        self.nerf.eval()

        val_metrics = defaultdict(float)
        base_tmp_path = None
        try:
            if 'RANK' in os.environ:
                base_tmp_path = Path(self.hparams.exp_name) / os.environ['TORCHELASTIC_RUN_ID']
                metric_path = base_tmp_path / 'tmp_val_metrics'
                image_path = base_tmp_path / 'tmp_val_images'

                world_size = int(os.environ['WORLD_SIZE'])
                indices_to_eval = np.arange(int(os.environ['RANK']), len(self.val_items), world_size)
                if self.is_master:
                    base_tmp_path.mkdir()
                    metric_path.mkdir()
                    image_path.mkdir()
                dist.barrier()
                
            else:
                indices_to_eval = np.arange(len(self.val_items))
                image_path = self.experiment_path / 'valimg'
                image_path.mkdir(exist_ok=True)

            for i in main_tqdm(indices_to_eval):
                with torch.no_grad():
                    metadata_item = self.val_items[i]
                    if metadata_item.is_depth_frame():
                        is_depth = True
                        viz_image = metadata_item.load_image().float()
                        assert viz_image.shape[-1] == 1 or len(viz_image.shape) == 2, f"depth image with shape {viz_image.shape}"
                    else:
                        is_depth = False
                        viz_image = metadata_item.load_image().float() / 255.
                        assert viz_image.shape[-1] == 3, f"color image with shape {viz_image.shape}"
                    img_w, img_h = viz_image.shape[0], viz_image.shape[1]

                with torch.inference_mode(mode=True):
                    results, _ = self.render_image(metadata_item)
                    typ = 'fine' if 'rgb_fine' in results else 'coarse'
                    viz_result_rgbs = results[f'rgb_{typ}'].view((img_w, img_h, 3)).cpu()
                    if not is_depth:
                        eval_rgbs = viz_image[:, viz_image.shape[1] // 2:].contiguous()
                        eval_result_rgbs = viz_result_rgbs[:, viz_image.shape[1] // 2:].contiguous()

                        val_psnr = psnr(eval_result_rgbs.view(-1, 3), eval_rgbs.view(-1, 3))

                        metric_key = 'val/psnr/{}'.format(i)
                        if self.writer is not None:
                            self.writer.add_scalar(metric_key, val_psnr, train_index)
                        else:
                            torch.save({'value': val_psnr, 'metric_key': metric_key, 'agg_key': 'val/psnr'},
                                        metric_path / 'psnr-{}.pt'.format(i))

                        val_metrics['val/psnr'] += val_psnr

                        val_ssim = ssim(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs, 1)

                        metric_key = 'val/ssim/{}'.format(i)
                        if self.writer is not None:
                            self.writer.add_scalar(metric_key, val_ssim, train_index)
                        else:
                            torch.save({'value': val_ssim, 'metric_key': metric_key, 'agg_key': 'val/ssim'},
                                        metric_path / 'ssim-{}.pt'.format(i))

                        val_metrics['val/ssim'] += val_ssim

                        val_lpips_metrics = lpips(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs)

                        for network in val_lpips_metrics:
                            agg_key = 'val/lpips/{}'.format(network)
                            metric_key = '{}/{}'.format(agg_key, i)
                            if self.writer is not None:
                                self.writer.add_scalar(metric_key, val_lpips_metrics[network], train_index)
                            else:
                                torch.save(
                                    {'value': val_lpips_metrics[network], 'metric_key': metric_key, 'agg_key': agg_key},
                                    metric_path / 'lpips-{}-{}.pt'.format(network, i))

                            val_metrics[agg_key] += val_lpips_metrics[network]
                    else:
                        self.writer.add_histogram('val/weights_dist', torch.log(results['weights_fine']), global_step=self.iter_step)
                        depths_metric, render_depth_metric = viz_image * self.pose_scale_factor, results[f'depth_{typ}'].view(-1) * self.pose_scale_factor
                        depths_metric = depths_metric.view(-1)
                        render_depth_metric = render_depth_metric[depths_metric > 1e-5]
                        depths_metric = depths_metric[depths_metric > 1e-5]
                        depth_metrics = {
                            "val_depth/RMSE/{}".format(i): depth_rmse(render_depth_metric, depths_metric),
                            "val_depth/RMSE_log/{}".format(i): depth_rmse_log(render_depth_metric, depths_metric),
                            "val_depth/Abs_Rel/{}".format(i): depth_abs_rel(render_depth_metric, depths_metric),
                            "val_depth/Sq_Rel/{}".format(i): depth_sq_rel(render_depth_metric, depths_metric),
                            "val_depth/δ_1/{}".format(i): depth_delta(render_depth_metric, depths_metric, 1),
                            "val_depth/δ_2/{}".format(i): depth_delta(render_depth_metric, depths_metric, 2),
                            "val_depth/δ_3/{}".format(i): depth_delta(render_depth_metric, depths_metric, 3),
                        }
                        for key, value in depth_metrics.items():
                            if self.writer is not None:
                                self.writer.add_scalar(key, value, train_index)
                            else:
                                torch.save({'value': value, 'metric_key': key, 'agg_key': key},
                                        metric_path / 'depth-{}.pt'.format(i))

                            val_metrics[key] += value

                        metric_key = 'val_pose/rotation/{}'.format(i)
                        if self.writer is not None:
                            self.writer.add_scalar(metric_key, results['rot_mse'], train_index)
                        val_metrics['val/rotation'] += results['rot_mse']

                        metric_key = 'val_pose/translation/{}'.format(i)
                        if self.writer is not None:
                            self.writer.add_scalar(metric_key, results['trans_mse'], train_index)
                        val_metrics['val/translation'] += results['trans_mse']

                    viz_result_rgbs = viz_result_rgbs.view(viz_image.shape[0], viz_image.shape[1], 3).cpu()
                    viz_depth = results[f'depth_{typ}']
                    if f'fg_depth_{typ}' in results:
                        to_use = results[f'fg_depth_{typ}'].view(-1)
                        while to_use.shape[0] > 2 ** 24:
                            to_use = to_use[::2]
                        ma = torch.quantile(to_use, 0.95)

                        viz_depth = viz_depth.clamp_max(ma)

                    img = Runner._create_result_image(viz_image if not is_depth else None, viz_result_rgbs,
                                                      viz_image if is_depth else None, viz_depth)


                    save_path = self.experiment_path / 'valimg' / f'val-{self.iter_step}'
                    if not os.path.exists(save_path):
                        save_path.mkdir()
                    img.save(str(save_path / '{}.jpg'.format(i)))
                    if self.writer is not None:
                        self.writer.add_image(('val_depth/{}' if is_depth else 'val_rgb/{}').format(i), T.ToTensor()(img), train_index)

                    del results

            if 'RANK' in os.environ:
                dist.barrier()
                if self.writer is not None:
                    for metric_file in metric_path.iterdir():
                        metric = torch.load(metric_file, map_location='cpu')
                        self.writer.add_scalar(metric['metric_key'], metric['value'], train_index)
                        val_metrics[metric['agg_key']] += metric['value']
                    for image_file in image_path.iterdir():
                        img = Image.open(str(image_file))
                        self.writer.add_image('val/{}'.format(image_file.stem), T.ToTensor()(img), train_index)


                dist.barrier()

            self.nerf.train()
        finally:
            if self.is_master and base_tmp_path is not None:
                shutil.rmtree(base_tmp_path)

        return val_metrics

    def _save_checkpoint(self, optimizers: Dict[str, any], scaler: GradScaler, train_index: int, dataset_index: int,
                         dataset_state: Optional[str]) -> None:
        dict = {
            'model_state_dict': self.nerf.state_dict(),
            'scaler': scaler.state_dict(),
            'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
            'iteration': train_index,
            'torch_random_state': torch.get_rng_state(),
            'np_random_state': np.random.get_state(),
            'random_state': random.getstate(),
            'dataset_index': dataset_index
        }

        if self.pose_correction is not None:
            dict['pose_correction_state_dict'] = self.pose_correction.state_dict()

        if dataset_state is not None:
            dict['dataset_state'] = dataset_state

        if self.bg_nerf is not None:
            dict['bg_model_state_dict'] = self.bg_nerf.state_dict()

        torch.save(dict, self.model_path / '{}.pt'.format(train_index))

    def render_image(self, metadata: ImageMetadata) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        with torch.no_grad():
            directions = get_ray_directions(metadata.W,
                                            metadata.H,
                                            metadata.intrinsics[0],
                                            metadata.intrinsics[1],
                                            metadata.intrinsics[2],
                                            metadata.intrinsics[3],
                                            self.hparams.center_pixels,
                                            self.device)

        with torch.cuda.amp.autocast(enabled=self.hparams.amp):
            c2w = metadata.c2w.to(self.device)
            gt_c2w = metadata.gt_pose.to(self.device)
            if self.pose_correction is not None and metadata.is_depth:
                c2w = self.pose_correction.forward_c2w(metadata.image_index, c2w)
            else:
                c2w = pp.mat2SE3(c2w.double()).float()
            gt_c2w = pp.mat2SE3(gt_c2w.double()).float()
            metrics = {}
            if self.hparams.have_gt_poses:
                error_trans_axes = (c2w.translation() - gt_c2w.translation()).abs().cpu().numpy() * self.pose_scale_factor
                error_trans = np.linalg.norm(error_trans_axes)
                theta_rot = (torch.acos(torch.sum(c2w.rotation().tensor() * gt_c2w.rotation().tensor(), dim=-1).abs().clamp(-1, 1)) * 360 / math.pi).cpu().numpy()
                metrics = {
                    'rot_mse': theta_rot,
                    'trans_mse': error_trans,
                }
            rays = get_rays(directions, c2w.matrix()[:3, :], self.near, self.far, self.ray_altitude_range)

            rays = rays.view(-1, 8).to(self.device, non_blocking=True)  # (H*W, 8)
            image_indices = metadata.image_index * torch.ones(rays.shape[0], device=rays.device) \
                if self.hparams.appearance_dim > 0 else None
            results = {}

            if 'RANK' in os.environ:
                nerf = self.nerf.module
            else:
                nerf = self.nerf

            if self.bg_nerf is not None and 'RANK' in os.environ:
                bg_nerf = self.bg_nerf.module
            else:
                bg_nerf = self.bg_nerf

            for i in range(0, rays.shape[0], self.hparams.image_pixel_batch_size):
                result_batch, _ = render_rays(nerf=nerf, bg_nerf=bg_nerf,
                                              rays=rays[i:i + self.hparams.image_pixel_batch_size],
                                              image_indices=image_indices[
                                                            i:i + self.hparams.image_pixel_batch_size] if self.hparams.appearance_dim > 0 else None,
                                              hparams=self.hparams,
                                              sphere_center=self.sphere_center,
                                              sphere_radius=self.sphere_radius,
                                              get_depth=True,
                                              get_depth_variance=False,
                                              get_bg_fg_rgb=True,
                                              progress=1,
                                              )

                with torch.no_grad():
                    for key, value in result_batch.items():
                        if key not in results:
                            results[key] = []

                        results[key].append(value.cpu())
                del result_batch
                torch.cuda.empty_cache()

            for key, value in results.items():
                results[key] = torch.cat(value)

            for key, value in metrics.items():
                results[key] = torch.tensor(value)

            return results, rays

    @staticmethod
    def _create_result_image(rgbs: torch.Tensor, result_rgbs: torch.Tensor, gt_depth: torch.Tensor
                            , result_depths: torch.Tensor) -> Image:
        if gt_depth is not None:
            depth_vis = Runner.visualize_scalars(torch.log(result_depths + 1e-8).view(gt_depth.shape[0], gt_depth.shape[1]).cpu())
            gt_depth_vis = Runner.visualize_scalars(torch.log(gt_depth + 1e-8).view(gt_depth.shape[0], gt_depth.shape[1]).cpu())
        else:
            depth_vis = Runner.visualize_scalars(torch.log(result_depths + 1e-8).view(rgbs.shape[0], rgbs.shape[1]).cpu())
            gt_depth_vis = torch.zeros_like(torch.tensor(depth_vis))
        depth = (gt_depth_vis, depth_vis)
        if rgbs is not None:
            images = (rgbs * 255, result_rgbs * 255)
        else:
            images = (torch.zeros_like(torch.tensor(result_rgbs)), result_rgbs * 255)
        ret = np.concatenate([np.concatenate(images, axis=1), np.concatenate(depth, axis=1)], axis=0).astype(np.uint8)
        return Image.fromarray(ret)

    @staticmethod
    def visualize_scalars(scalar_tensor: torch.Tensor) -> np.ndarray:
        to_use = scalar_tensor.view(-1)
        while to_use.shape[0] > 2 ** 24:
            to_use = to_use[::2]

        mi = torch.quantile(to_use, 0.05)
        ma = torch.quantile(to_use, 0.95)
        scalar_tensor = (scalar_tensor - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
        scalar_tensor = scalar_tensor.clamp_(0, 1)

        scalar_tensor = ((1 - scalar_tensor) * 255).byte().numpy()  # inverse heatmap
        return cv2.cvtColor(cv2.applyColorMap(scalar_tensor, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)

    def _get_image_metadata(self) -> Tuple[List[ImageMetadata], List[ImageMetadata]]:
        """
        从 hparams.dataset_path 指定的数据集位置读入元信息
        """
        dataset_path = Path(self.hparams.dataset_path)
        depth_track = self.hparams.depth_track

        train_path_candidates_rgb = sorted(list((dataset_path / 'train' / 'metadata_rgb').iterdir()))
        train_paths_rgb = [train_path_candidates_rgb[i] for i in
                           range(0, len(train_path_candidates_rgb), self.hparams.train_every)]
        train_path_candidates_depth = sorted(list((dataset_path / 'train' / f'metadata_depth_{depth_track}').iterdir()))
        train_paths_depth = [train_path_candidates_depth[i] for i in
                             range(0, len(train_path_candidates_depth), self.hparams.train_every)]

        val_paths_rgb = sorted(list((dataset_path / 'val' / 'metadata_rgb').iterdir()))
        train_paths_rgb += val_paths_rgb
        train_paths_rgb.sort(key=lambda x: x.name)
        val_paths_set_rgb = set(val_paths_rgb)

        val_paths_depth = sorted(list((dataset_path / 'val' / f'metadata_depth_{depth_track}').iterdir()))
        train_paths_depth += val_paths_depth
        train_paths_depth.sort(key=lambda x: x.name)
        val_paths_set_depth = set(val_paths_depth)

        self.n_depth_frames = len(train_paths_depth)

        image_indices = {}
        for i, train_path in enumerate(train_paths_rgb):
            image_indices[train_path.name] = i

        depth_indices = {}
        for i, train_path in enumerate(train_paths_depth):
            depth_indices[train_path.name] = i

        """
        - see: https://github.com/cmusatyalab/mega-nerf/issues/18 for why validation set images are still in the training set
        """
        train_items = [
            self._get_metadata_item(x, depth_track, image_indices[x.name], self.hparams.train_scale_factor, x in val_paths_set_rgb, False) for x
            in train_paths_rgb] + [
            self._get_metadata_item(x, depth_track, depth_indices[x.name], self.hparams.train_scale_factor, x in val_paths_set_depth, True) for x
            in train_paths_depth
            ]
        val_items = [self._get_metadata_item(x, depth_track, image_indices[x.name], self.hparams.val_scale_factor, True, False) for x in val_paths_rgb] + \
                    [self._get_metadata_item(x, depth_track, depth_indices[x.name], self.hparams.val_scale_factor, True, True) for x in val_paths_depth]

        return train_items, val_items

    def _get_metadata_item(self, metadata_path: Path, depth_name: str, image_index: int, scale_factor: int,
                           is_val: bool, is_depth: bool) -> ImageMetadata:
        gt_pose_path = None
        if not is_depth:
            image_path = None
            for extension in ['.jpg', '.JPG', '.png', '.PNG']:
                candidate = metadata_path.parent.parent / 'rgbs' / '{}{}'.format(metadata_path.stem, extension)
                if candidate.exists():
                    image_path = candidate
                    break
            assert image_path is not None
        else:
            depth_path = None
            for extension in ['.jpg', '.JPG']:
                candidate = metadata_path.parent.parent / f'depthvis_{depth_name}' / '{}{}'.format(metadata_path.stem, extension)
                if candidate.exists():
                    depth_path = candidate
                    break
            gt_pose_path = metadata_path.parent.parent / f'pose_gt_{depth_name}' / '{}{}'.format(metadata_path.stem, '.txt')
            assert depth_path is not None, metadata_path.parent.parent / f'depthvis_{depth_name}' / '{}{}'.format(metadata_path.stem, extension)
            assert not self.hparams.have_gt_poses or gt_pose_path.exists()
            image_path = depth_path

        metadata = torch.load(metadata_path, map_location='cpu')
        intrinsics = metadata['intrinsics'] / scale_factor
        # assert metadata['W'] % scale_factor == 0
        # assert metadata['H'] % scale_factor == 0
        try:
            pp.mat2SE3(metadata['c2w'])
        except all:
            raise ValueError()

        color_mask_path = Path(self.hparams.cluster_mask_path) / 'mask_rgb' / f'{self.hparams.centroid}'
        depth_mask_path = Path(self.hparams.cluster_mask_path) / f'mask_depth_{depth_name}' / f'{self.hparams.centroid}'
        cluster_mask_path = depth_mask_path if is_depth else color_mask_path
        if self.hparams.cluster_mask_path is not None:
            if image_index == 0:
                main_print('Using cluster mask path: {}'.format(cluster_mask_path))
            mask_path = cluster_mask_path / metadata_path.name
        else:
            mask_path = None

        return ImageMetadata(image_path, metadata['c2w'] if is_depth else metadata['c2w'], metadata['W'] // scale_factor, metadata['H'] // scale_factor,
                             intrinsics, image_index, None if (is_val and self.hparams.all_val) else mask_path, is_val, self.pose_scale_factor, is_depth, 
                             gt_pose_path if self.hparams.have_gt_poses else None)

    def _get_experiment_path(self) -> Path:
        exp_dir = Path(self.hparams.exp_name)
        exp_dir.mkdir(parents=True, exist_ok=True)
        existing_versions = [int(x.name) for x in exp_dir.iterdir()]
        version = 0 if len(existing_versions) == 0 else max(existing_versions) + 1
        experiment_path = exp_dir / str(version)
        return experiment_path
