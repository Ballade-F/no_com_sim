#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys

import numpy as np
import math
from copy import deepcopy
from typing import List
import random

if __name__ == '__main__':
    script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(script_path)


# class Pose2D:  
#     def __init__(self, x:float = 0, y:float = 0, theta:float = 0):
#         self.x:float = x
#         self.y:float = y
#         self.theta:float = theta
    
#     def __str__(self) -> str:
#         return "Pose2D: (%f, %f, %f)" % (self.x, self.y, self.theta)

#     def __repr__(self):
#         return self.__str__()

#     def __add__(self, other:Pose2D):
#         return Pose2D(x = self.x + other.x, y = self.y + other.y)

# 贪心算法任务规划器
class GreedyTaskAllocationPlanner:
    def __init__(self):
        self.time_estimate_func:function = self.pose_distance
        self.eval_time_estimate_func:function = self.pose_distance
        self.reward_time_decay = 1.0

    def config_time_estimate_function(self, func:function):
        self.time_estimate_func = func

    def config_eval_time_estimate_function(self, func:function):
        self.eval_time_estimate_func = func

    def config_reward_time_decay(self, decay:float):
        self.reward_time_decay = decay

    # 贪心算法任务分配函数
    # agent_poses: 智能体初始位置
    # task_poses:  任务点位置
    def greedy_allocate(self, agent_poses:np.ndarray, task_poses:np.ndarray) -> List[List[int]]:
        # 智能体个数与任务个数
        num_agent = agent_poses.shape[0]
        num_task = task_poses.shape[0]
        # 智能体到任务距离矩阵
        agent_task_dist_mat = np.zeros(shape = (num_agent, num_task), dtype = np.float64)
        for i_agent in range(num_agent):
            for i_task in range(num_task):
                dist = self.time_estimate_func(agent_poses[i_agent], task_poses[i_task])
                agent_task_dist_mat[i_agent, i_task] = dist
        # 任务之间距离矩阵
        task_dist_mat = np.zeros(shape = (num_task, num_task), dtype = np.float64)
        for i_task1 in range(num_task):
            for i_task2 in range(i_task1):
                dist = self.time_estimate_func(task_poses[i_task1], task_poses[i_task2])
                task_dist_mat[i_task1, i_task2] = dist
                task_dist_mat[i_task2, i_task1] = dist
        # 距离矩阵
        dist_mat = deepcopy(agent_task_dist_mat)
        
        # 智能体任务序列
        agent_schedules = [[] for i_agent in range(num_agent)]
        # 智能体已有路径长度
        agent_path_lengths = [0 for i_agent in range(num_agent)]
        for i_task in range(num_task):
            # 距离矩阵求最小值并计算对应任务编号与智能体编号
            min_dist_index = np.argmin(dist_mat)
            min_dist_agent = int(min_dist_index / num_task)
            min_dist_task = int(min_dist_index % num_task)
            # 距离矩阵对应列设为inf 任务不参与后续分配
            dist_mat[:, min_dist_task] = np.inf
            # 智能体任务序列添加
            agent_schedules[min_dist_agent].append(min_dist_task)
            # 智能体已有路径长度更新
            if len(agent_schedules[min_dist_agent]) == 1:
                agent_path_lengths[min_dist_agent] += agent_task_dist_mat[min_dist_agent, min_dist_task]
            else:
                last_task = agent_schedules[min_dist_agent][-2]
                agent_path_lengths[min_dist_agent] += task_dist_mat[last_task, min_dist_task]
            # 新分配获得任务智能体更新距离矩阵对应行 设为已有长度+从当前点到下一点的距离
            for i_task in range(num_task):
                if dist_mat[min_dist_agent, i_task] == np.inf:
                    continue
                dist_mat[min_dist_agent, i_task] = agent_path_lengths[min_dist_agent]
                dist_mat[min_dist_agent, i_task] += task_dist_mat[min_dist_task, i_task]
        return agent_schedules

    # 任务分配距离代价评估函数
    # agent_poses: 智能体初始位置
    # task_poses:  任务点位置
    # schedules:   智能体任务序列
    def allocation_distance_eval(self, agent_poses:np.ndarray, task_poses:np.ndarray,
                                  schedules:List[list]):
        num_agent = agent_poses.shape[0]
        num_task = task_poses.shape[0]
        agent_distances = [0 for i_agent in range(num_agent)]
        for i_agent in range(num_agent):
            agent_pose = agent_poses[i_agent]
            for task in schedules[i_agent]:
                agent_distances[i_agent] += self.time_estimate_func(agent_pose, task_poses[task])
                agent_pose = task_poses[task]
        total_distance = 0
        for i_agent in range(num_agent):
            total_distance += agent_distances[i_agent]
        return total_distance
    
    # 任务分配距离代价评估函数
    # agent_poses: 智能体初始位置
    # task_poses:  任务点位置
    # schedules:   智能体任务序列
    # reward_decay:收益衰减系数
    def allocation_reward_eval(self, agent_poses:np.ndarray, task_poses:np.ndarray,
            schedules:List[list]):
        num_agent = agent_poses.shape[0]
        num_task = task_poses.shape[0]
        agent_rewards = [0 for i_agent in range(num_agent)]
        for i_agent in range(num_agent):
            agent_pose = agent_poses[i_agent]
            agent_time = 0
            for task in schedules[i_agent]:
                agent_time += self.eval_time_estimate_func(agent_pose, task_poses[task])
                agent_rewards[i_agent] += math.exp(-self.reward_time_decay * agent_time)
                agent_pose = task_poses[task]
        total_reward = 0
        for i_agent in range(num_agent):
            total_reward += agent_rewards[i_agent]
        return total_reward

    # 距离计算函数
    # start_pose: 起点位置
    # goal_pose:  终点位置
    def pose_distance(self, start_pose:np.ndarray, goal_pose:np.ndarray):
        dx = start_pose[0] - goal_pose[0]
        dy = start_pose[1] - goal_pose[1]
        return math.sqrt(dx*dx + dy*dy)
        # dx = start_pose.x - goal_pose.x
        # dy = start_pose.y - goal_pose.y
        # return math.sqrt(dx*dx + dy*dy)


    # 贪心算法任务分配函数
    # at_mat: 智能体到任务距离矩阵(num_agent, num_task)
    # task_mat:  任务之间距离矩阵
    def greedy_allocate_mat(self, at_mat:np.ndarray, task_mat:np.ndarray) -> List[List[int]]:
        # 距离矩阵
        dist_mat = deepcopy(at_mat)
        num_agent = at_mat.shape[0]
        num_task = task_mat.shape[0]
        
        # 智能体任务序列
        agent_schedules = [[] for i_agent in range(num_agent)]
        # 智能体已有路径长度
        agent_path_lengths = [0 for i_agent in range(num_agent)]
        for i_task in range(num_task):
            # 距离矩阵求最小值并计算对应任务编号与智能体编号
            min_dist_index = np.argmin(dist_mat)
            min_dist_agent = int(min_dist_index / num_task)
            min_dist_task = int(min_dist_index % num_task)
            # 距离矩阵对应列设为inf 任务不参与后续分配
            dist_mat[:, min_dist_task] = np.inf
            # 智能体任务序列添加
            agent_schedules[min_dist_agent].append(min_dist_task)
            # 智能体已有路径长度更新
            if len(agent_schedules[min_dist_agent]) == 1:
                agent_path_lengths[min_dist_agent] += at_mat[min_dist_agent, min_dist_task]
            else:
                last_task = agent_schedules[min_dist_agent][-2]
                agent_path_lengths[min_dist_agent] += task_mat[last_task, min_dist_task]
            # 新分配获得任务智能体更新距离矩阵对应行 设为已有长度+从当前点到下一点的距离
            for i_task in range(num_task):
                if dist_mat[min_dist_agent, i_task] == np.inf:
                    continue
                dist_mat[min_dist_agent, i_task] = agent_path_lengths[min_dist_agent]
                dist_mat[min_dist_agent, i_task] += task_mat[min_dist_task, i_task]
        return agent_schedules

    # 任务分配距离代价评估函数
    # at_mat: 智能体到任务距离矩阵(num_agent, num_task)
    # task_mat:  任务之间距离矩阵
    # schedules:   智能体任务序列
    def allocation_distance_eval_mat(self, at_mat:np.ndarray, task_mat:np.ndarray,
                                  schedules:List[list]):
        num_agent = at_mat.shape[0]
        num_task = task_mat.shape[0]
        agent_distances = [0 for i_agent in range(num_agent)]
        for i_agent in range(num_agent):
            if len(schedules[i_agent]) == 0:
                continue
            agent_distances[i_agent] += at_mat[i_agent, schedules[i_agent][0]]
            for i in range(1, len(schedules[i_agent])):
                agent_distances[i_agent] += task_mat[schedules[i_agent][i-1], schedules[i_agent][i]]
        total_distance = 0
        for i_agent in range(num_agent):
            total_distance += agent_distances[i_agent]
        return total_distance


if __name__ == '__main__':
    planner = GreedyTaskAllocationPlanner()
    num_task = 20
    num_agent = 3
    task_poses = np.random.rand(num_task, 2)
    agent_poses = np.random.rand(num_agent, 2)
    # task_poses = []
    # num_task = 20
    # for i_task in range(num_task):
    #     task_poses.append(Pose2D(x = random.random(), y = random.random()))
    # agent_poses = []
    # num_agent = 5
    # for i_agent in range(num_agent):
    #     agent_poses.append(Pose2D(x = random.random(), y = random.random()))

    planner.config_time_estimate_function(planner.pose_distance)
    agent_schedules2 = planner.greedy_allocate(agent_poses, task_poses)
    print("agent schedules: ")
    print(agent_schedules2)
    print("total distance: ")
    print(planner.allocation_distance_eval(agent_poses, task_poses, agent_schedules2))
    print("total reward: ")
    print(planner.allocation_reward_eval(agent_poses, task_poses, agent_schedules2))