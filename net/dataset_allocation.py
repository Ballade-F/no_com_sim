import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import utils.map as mp


        

#出于批量训练的考虑，一个batch中的规模是固定的，需要掩盖填充不足一个batch的数据
class AllocationDataset(Dataset):
    def __init__(self, n_batch, batch_size, seed, n_robot_min=3, n_robot_max=10, n_task_min=10, n_task_max=30, n_obstacle_min=5, n_obstacle_max=10,
                 n_x: int = 100, n_y: int = 100, resolution_x: float = 0.1, resolution_y: float = 0.1):
        super(AllocationDataset, self).__init__()
        self.rng = np.random.default_rng(seed)
        self.n_batch = n_batch
        self.batch_size = batch_size
        self.n_robot_min = n_robot_min
        self.n_robot_max = n_robot_max
        self.n_task_min = n_task_min
        self.n_task_max = n_task_max
        self.n_obstacle_min = n_obstacle_min
        self.n_obstacle_max = n_obstacle_max
