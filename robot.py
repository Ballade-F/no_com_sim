import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils.map as mp
import utils.ringbuffer as ringbuffer
import pnc.dwa as dwa
import task_allocation.hungarian as hungarian
import task_allocation.greedy_allocation_lib as greedy
import task_allocation.ga_allocation_lib as ga
import net.mtsp as mtsp
import net.dataset_intention as dataset_intention
import net.intention_judgment as intention_judgment


class Robot:
    def __init__(self, robot_id, map:mp.Map, buffer_size:int ,device='cpu'):
    # 成员变量
        # 基本信息
        self.robot_id = robot_id
        self.map = map.deepcopy()
        self.device = device
        self.n_robot = map.n_starts
        self.n_task = map.n_tasks

        # 感知决策
        self.buffer_size = buffer_size
        self.buffer_robots = ringbuffer.RingBuffer(self.buffer_size)
        self.buffer_tasks = ringbuffer.RingBuffer(self.buffer_size)
        self.path = []
        self.target = []  # n_robot个元素，每个元素是一个任务的index

        
        # 控制

        # 调度用的标志量

        # 算法类实例与网络类实例


    # 初始化

        