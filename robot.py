import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy


import utils.map as mp
import utils.ringbuffer as ringbuffer
import pnc.dwa as dwa
import pnc.path_planner as path_planner
import task_allocation.hungarian as hungarian
import task_allocation.greedy_allocation_lib as greedy
import task_allocation.ga_allocation_lib as ga
import net.allocation as allocation
import net.intention as intention


class RobotData():
    def __init__(self, n_robot) -> None:
        self.n_robot = n_robot
        self.data = np.zeros((n_robot, 3))# x, y, theta

class TaskData():
    def __init__(self, n_task) -> None:
        self.n_task = n_task
        self.data = np.zeros((n_task, 3))# x, y, if_finished

class Robot:
    def __init__(self, robot_id, map:mp.Map, buffer_size:int ,device='cpu'):
    # 成员变量
        # 基本信息
        self.robot_id = robot_id
        self.map = deepcopy(map)
        self.device = device
        self.n_robot = map.n_starts
        self.n_task = map.n_tasks

        # 感知决策
        self.robot_data = RobotData(self.n_robot)
        self.task_data = TaskData(self.n_task)
        self.buffer_size = buffer_size
        self.buffer_robots = ringbuffer.RingBuffer(self.buffer_size)
        self.buffer_tasks = ringbuffer.RingBuffer(self.buffer_size)
        self.buffer_decisions = ringbuffer.RingBuffer(self.buffer_size)
        self.path = []
        self.target = []  # n_robot个元素，每个元素是一个任务的index

        
        # 控制
        
        self.w = 0
        self.v = 0

        # 调度用的标志量
        self.base_T = 0.1  # 最小调度周期，控制与感知周期
        self.keyframe_counter = 5  # 关键帧周期数
        self.decision_counter = 25  # 决策周期数

        self.counter_base = 0


    # 算法类实例与网络类实例
        self.controller = dwa.DWA()
        self.path_planner = path_planner.AStarPlanner(self.map.map, self.map.resolution_x, self.map.resolution_y)
        #TODO: 换成网络
        self.task_allocation = greedy.GreedyTaskAllocationPlanner()
        self.intention_judgment = intention.IntentionNet()

    # 初始化



# 最小周期回调函数
    def base_callback(self):
        self.counter_base += 1
        if self.counter_base == self.keyframe_counter:
            self._keyframe_callback()
        if self.counter_base == self.decision_counter:
            self._decision_callback()
            self.counter_base = 0
        self._control_callback()


    def _control_callback(self):
        pass


    def _keyframe_callback(self):
        pass

        
    def _decision_callback(self):
        pass