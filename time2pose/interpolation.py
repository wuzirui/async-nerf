import math
import torch
import numpy as np
from scipy import interpolate
import pypose as pp


class Interpolation1d:
    def __init__(self, dataset, method):
        self.method = method
        self.dataset = dataset
        self.models = self._interpolation1d()

    def _interpolation1d(self):
        train_data = self.dataset
        raw_x = np.concatenate(np.array(train_data.dataset.frames))
        x = np.array(train_data.dataset.poses_SE3.translation()[:, 0])
        y = np.array(train_data.dataset.poses_SE3.translation()[:, 1])
        z = np.array(train_data.dataset.poses_SE3.translation()[:, 2])
        rotation = train_data.dataset.poses_SE3.rotation()
        theta1 = np.array([rotation[i].euler()[0] for i in range(len(rotation))])
        theta2 = np.array([rotation[i].euler()[1] for i in range(len(rotation))])
        theta3 = np.array([rotation[i].euler()[2] for i in range(len(rotation))])
        f_x = interpolate.interp1d(raw_x, x, kind=self.method)
        f_y = interpolate.interp1d(raw_x, y, kind=self.method)
        f_z = interpolate.interp1d(raw_x, z, kind=self.method)
        f_theta1 = interpolate.interp1d(raw_x, np.array(theta1), kind=self.method)
        f_theta2 = interpolate.interp1d(raw_x, np.array(theta2), kind=self.method)
        f_theta3 = interpolate.interp1d(raw_x, np.array(theta3), kind=self.method)
        return [f_x, f_y, f_z, f_theta1, f_theta2, f_theta3]

    def get_output(self, timestamps):
        timestamps = np.concatenate(np.array(timestamps.cpu()))
        pred_x = self.models[0](timestamps)
        pred_y = self.models[1](timestamps)
        pred_z = self.models[2](timestamps)
        pred_theta1 = self.models[3](timestamps)
        pred_theta2 = self.models[4](timestamps)
        pred_theta3 = self.models[5](timestamps)
        pred_trans = torch.tensor(np.array([pred_x, pred_y, pred_z])).T
        pred_rot = pp.euler2SO3(torch.tensor(np.stack([pred_theta1, pred_theta2, pred_theta3])).T).tensor()
        return pred_trans, pred_rot

