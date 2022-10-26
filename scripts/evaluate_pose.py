import argparse
from pathlib import Path
import torch
import numpy as np
import math
import pypose as pp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--reference_traj', type=str, help='path to the reference trajectory poses')
    parser.add_argument('--evaluate_traj', type=str, help='path to the trajectory under evaluation')
    parser.add_argument('--allow_misalign', default=False, action='store_true', help='find nearest reference frame when frame timestamps in evaluation traj not found in reference traj')
    parser.add_argument('--pose_scale_factor', type=float, default=1)

    return parser.parse_known_args()[0]

hparams = _get_opts()
reference_filenames = sorted([x for x in Path(hparams.reference_traj).iterdir() if x.suffix == '.txt'], key=lambda x: float(x.stem))
evaluate_filenames = sorted([x for x in Path(hparams.evaluate_traj).iterdir() if x.suffix == '.txt'], key=lambda x: float(x.stem))
evaluate_poses = [np.loadtxt(x) for x in evaluate_filenames]
evaluate_poses = pp.mat2SE3(evaluate_poses)

if not hparams.allow_misalign:
    assert len(reference_filenames) == len(evaluate_filenames)

reference_poses = []
ref_dir = Path(hparams.reference_traj)
ref_files = [x for x in ref_dir.iterdir() if x.suffix == '.txt']
for x in evaluate_filenames:
    pose_filename = ref_dir / x.name
    if pose_filename.exists() or not hparams.allow_misalign:
        pose = np.loadtxt(pose_filename)
    else:
        l = np.array([abs(float(y.stem) - float(x.stem)) for y in ref_files])
        pose_path = ref_files[np.argmin(l)]
        print(f'{x.stem} estimates to {pose_path.stem}')
        pose = np.loadtxt(pose_path)
    reference_poses.append(pose)
reference_poses = pp.mat2SE3(reference_poses)
    
pred_trans = evaluate_poses.translation().to(device)
pred_rot = evaluate_poses.rotation().to(device)
gt_SE3 = reference_poses.to(device)
error_trans_axes = (pred_trans - gt_SE3.translation()).abs().cpu().numpy() * hparams.pose_scale_factor
error_trans = np.linalg.norm(error_trans_axes, axis=-1)
theta_rot = (torch.acos(torch.sum(pred_rot.tensor() * gt_SE3.rotation().tensor(), dim=-1).abs().clamp(-1, 1)) * 360 / math.pi).cpu().numpy()
error_trans_axes_median = np.median(error_trans_axes, axis=0)
error_trans_axes_mean = np.mean(error_trans_axes, axis=0)
error_trans_median = np.median(error_trans, axis=0)
error_trans_mean = np.mean(error_trans, axis=0)
theta_rot_median = np.median(theta_rot, axis=0)
theta_rot_mean = np.mean(theta_rot, axis=0)
print({
    'error_trans_median': error_trans_median,
    'error_trans_mean': error_trans_mean,
    'theta_median': theta_rot_median,
    'theta_mean': theta_rot_mean,
})
