import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from pytorch3d import transforms
from torch.autograd import Variable
from tqdm import tqdm
from pathlib import Path
from tensorboardX import SummaryWriter

# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# torch.cuda.manual_seed_all(1)

exp_folder = Path('exp')
exp_name = str(max(int(name.name) for name in exp_folder.iterdir()) + 1)
writer = SummaryWriter(f'exp/{exp_name}')

pose_filename = 'manmade_12.txt'
pose_format = 'kitti'
valset_interval = 10
logging.basicConfig(level=logging.DEBUG  # 设置日志输出格式
                    # ,filename="demo.log" #log日志输出的文件位置和文件名
                    # ,filemode="w" #文件的写入格式，w为重新写入文件，默认是追加
                    , format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s"
                    # 日志输出的格式
                    # -8表示占位符，让输出左对齐，输出长度都为8位
                    , datefmt="%Y-%m-%d %H:%M:%S"  # 时间输出的格式
                    )
logger = logging.getLogger()


def read_pose_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    if pose_format == 'kitti':
        pose = [([eval(num) for num in line.split()]) for line in lines]
        pose = np.array(pose, dtype=np.float32).reshape((-1, 12))
    return pose


class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.net2 = nn.Sequential(
            nn.Linear(257, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            nn.Linear(64, 12),
        )

    def forward(self, t):
        mid = self.net1(t)
        mid = torch.cat([mid, t], -1)
        return self.net2(mid)


def compute_error_in_q(epoch, labels, results):
    result_list = []
    for i in range(labels.shape[0]):
        predict_pose = results[i]
        pose = labels[i].reshape((-1, 3, 4)).cpu().numpy()
        R_torch = predict_pose.reshape((-1, 3, 4))[:, :3, :3]
        predict_pose = predict_pose.reshape((-1, 3, 4)).cpu().numpy()
        # R = predict_pose[:, :3, :3]
        # res = R @ np.linalg.inv(R)
        u, s, v = torch.svd(R_torch)
        Rs = torch.matmul(u, v.transpose(-2, -1))
        predict_pose[:, :3, :3] = Rs[:, :3, :3].cpu().numpy()
        pose_q = transforms.matrix_to_quaternion(torch.Tensor(pose[:, :3, :3]))
        pose_x = pose[:, :3, 3]
        predicted_q = transforms.matrix_to_quaternion(torch.Tensor(predict_pose[:, :3, :3]))
        predicted_x = predict_pose[:, :3, 3]
        pose_q = pose_q.squeeze()
        pose_x = pose_x.squeeze()
        predicted_q = predicted_q.squeeze()
        predicted_x = predicted_x.squeeze()
        q1 = pose_q / torch.linalg.norm(pose_q)
        q2 = predicted_q / torch.linalg.norm(predicted_q)

        d = torch.abs(torch.sum(torch.matmul(q1, q2)))
        d = torch.clamp(d, -1., 1.)  # acos can only input [-1~1]
        theta = (2 * torch.acos(d) * 180 / math.pi).numpy()
        error_x = torch.linalg.norm(torch.Tensor(pose_x - predicted_x)).numpy()
        result_list.append([error_x, theta])
        # print('Iteration: {} Error XYZ (m): {} Error Q (degrees): {}'.format(epoch, error_x, theta))
    median_result = np.median(result_list, axis=0)
    mean_result = np.mean(result_list, axis=0)
    writer.add_scalar('eval/median_x', median_result[0], global_step=epoch)
    writer.add_scalar('eval/median_q', median_result[1], global_step=epoch)
    writer.add_scalar('eval/mean_x', mean_result[0], global_step=epoch)
    writer.add_scalar('eval/mean_q', mean_result[1], global_step=epoch)
    return

device = 'cuda:0'
pose = read_pose_file(pose_filename)
n_frames = len(pose)
ts = np.linspace(0, n_frames - 1, n_frames)
pose, ts = torch.tensor(pose, dtype=torch.float32).to(device), torch.tensor(ts, dtype=torch.float32).to(device)
ts = ts.view(-1, 1)
train_idx = [i for i in range(n_frames) if i % valset_interval != 0]
val_idx = [i for i in range(n_frames) if i % valset_interval == 0]
n_train = len(train_idx)
n_val = len(val_idx)

logger.info(f"frame num: {n_frames}, train set: {n_train}, val set: {n_val}")
train_ts, val_ts = ts[train_idx], ts[val_idx]
train_pose, val_pose = pose[train_idx], pose[val_idx]
network = PoseNet().to(device)
optimizer = torch.optim.Adam(list(network.parameters()), lr=5e-3)
train_epoch = 100000
val_interval = 2000
beta = 1


def pose_loss(predict_pose, pose, device, beta):
    loss_func = nn.MSELoss(reduction='mean').to(device)
    pose = pose.reshape((-1, 3, 4)).cpu().numpy()
    R_torch = predict_pose.reshape((-1, 3, 4))[:, :3, :3]
    predict_pose = predict_pose.reshape((-1, 3, 4)).cpu().detach().numpy()
    # R = predict_pose[:, :3, :3]
    # res = R @ np.linalg.inv(R)
    u, s, v = torch.svd(R_torch)
    Rs = torch.matmul(u, v.transpose(-2, -1))
    predict_pose[:, :3, :3] = Rs[:, :3, :3].cpu().detach().numpy()
    pose_q = transforms.matrix_to_quaternion(torch.Tensor(pose[:, :3, :3]))
    pose_x = pose[:, :3, 3]
    predicted_q = transforms.matrix_to_quaternion(torch.Tensor(predict_pose[:, :3, :3]))
    predicted_x = predict_pose[:, :3, 3]
    # q1 = pose_q / torch.linalg.norm(pose_q)
    # q2 = predicted_q / torch.linalg.norm(predicted_q)
    loss_x = loss_func(torch.Tensor(pose_x).to(device), torch.Tensor(predicted_x).to(device))
    loss_q = loss_func(torch.Tensor(pose_q).to(device), torch.Tensor(predicted_q).to(device))
    pose_loss = loss_x + beta * loss_q
    pose_loss = Variable(pose_loss.data, requires_grad=True)
    return pose_loss


bar = tqdm(range(train_epoch))
for epoch in bar:
    output = network(train_ts)
    optimizer.zero_grad()
    # loss = F.mse_loss(output, train_pose, reduction='mean')
    loss = pose_loss(output, train_pose, device, beta)
    loss.backward()
    optimizer.step()
    writer.add_scalar('train/loss', loss, global_step=epoch)

    bar.set_postfix_str(f'loss={loss}')
    if epoch % val_interval == 0 and epoch != 0:
        with torch.inference_mode(True):
            output = network(val_ts)
            compute_error_in_q(epoch, val_pose, output)
            # loss = F.mse_loss(output, val_pose, reduction='mean')
            loss = pose_loss(output, train_pose, device, beta)
            writer.add_scalar('val/loss', loss, global_step=epoch)
            print(f'validation loss = {loss}')

torch.save(network, "checkpoint.pt")
test_ts = ts
output = network(test_ts).cpu().detach().numpy()

output_filename = 'manmade_12_out.txt'
np.savetxt(output_filename, output)
