import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils.map as mp
import utils.ringbuffer as ringbuffer
import pnc.dwa as dwa
import pnc.path_planner as path_planner
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
        
        self.w = 0
        self.v = 0

        # 调度用的标志量
        self.base_T = 0.1  # 最小调度周期，控制与感知周期
        self.keyframe_mag = 5  # 关键帧周期数
        self.decision_mag = 25  # 决策周期数

        self.counter_base = 0


    # 算法类实例与网络类实例
        self.controller = dwa.DWA()
        self.path_planner = path_planner.AStarPlanner(self.map.map, self.map.resolution_x, self.map.resolution_y)
        self.task_allocation = mtsp.ActNet()
        self.intention_judgment = intention_judgment.IntentionNet()

    # 初始化



# 最小周期回调函数
    def base_callback(self):
        self.counter_base += 1
        if self.counter_base == self.keyframe_mag:
            self._keyframe_callback()
        if self.counter_base == self.decision_mag:
            self._decision_callback()
            self.counter_base = 0
        self._control_callback()


    def _control_callback(self):
        pass


    def _keyframe_callback(self):
        pass

        
    def _decision_callback(self):
        pass