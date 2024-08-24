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

# from basic_lib.basic_struct_lib import *
# from basic_lib.terminal_lib import *
class Pose2D:  
    def __init__(self, x:float = 0, y:float = 0, theta:float = 0):
        self.x:float = x
        self.y:float = y
        self.theta:float = theta
    
    def __str__(self) -> str:
        return "Pose2D: (%f, %f, %f)" % (self.x, self.y, self.theta)

    def __repr__(self):
        return self.__str__()

    def __add__(self, other:Pose2D):
        return Pose2D(x = self.x + other.x, y = self.y + other.y)

# 遗传算法任务规划器
class GATaskAllocationPlanner:
    # 基因型结构体
    class Gene:
        def __init__(self):
            # 任务序列基因型
            self.schedule_gene = []
            # 断点基因型
            self.breakpoint_gene = []
        
        def __init__(self, num_task:int, num_agent:int):
            self.schedule_gene = [i_task for i_task in range(num_task)]
            random.shuffle(self.schedule_gene)
            self.breakpoint_gene = [1 if i_task < (num_agent-1) else 0 for i_task in range(num_task+1)]
            random.shuffle(self.breakpoint_gene)
    
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
    
    # 遗传算法任务分配函数
    # agent_poses: 智能体初始位置
    # task_poses:  任务点位置
    def ga_allocate(self, agent_poses:List[Pose2D], task_poses:List[Pose2D]) -> List[List[int]]:
        self.num_agent = len(agent_poses)
        self.num_task = len(task_poses)
        self.agent_poses = deepcopy(agent_poses)
        self.task_poses = deepcopy(task_poses)
        
        print('Generate ETD matrix.')
        # 智能体到任务距离矩阵
        self.agent_task_dist_mat = np.zeros(shape = (self.num_agent, self.num_task), dtype = np.float64)
        for i_agent in range(self.num_agent):
            for i_task in range(self.num_task):
                dist = self.time_estimate_func(agent_poses[i_agent], task_poses[i_task])
                self.agent_task_dist_mat[i_agent, i_task] = dist
        # 任务之间距离矩阵
        self.task_dist_mat = np.zeros(shape = (self.num_task, self.num_task), dtype = np.float64)
        for i_task1 in range(self.num_task):
            for i_task2 in range(i_task1):
                dist = self.time_estimate_func(task_poses[i_task1], task_poses[i_task2])
                self.task_dist_mat[i_task1, i_task2] = dist
                dist = self.time_estimate_func(task_poses[i_task2], task_poses[i_task1])
                self.task_dist_mat[i_task2, i_task1] = dist
        print('Start GA optimization.')
        population_size = 30    # 种群规模
        max_iteration_num = 300 # 最大迭代次数

        # 初始种群
        population = [GATaskAllocationPlanner.Gene(self.num_task, self.num_agent)\
            for i in range(population_size)]
        for i_iteration in range(max_iteration_num):
            # 种群各父代繁殖产生子代构成子代种群
            child_population = []
            for gene in population:
                child_population = child_population + self.gene_reproduct(gene)
            # 种群选择，选择规模与父代规模相同
            population = self.population_filter(child_population, population_size)
            
            
        return self.gene_decode(population[0])
    
    # 种群筛选函数
    # population: 种群 基因型列表
    # num:        目标种群规模
    def population_filter(self, population:List[Gene], num:int):
        # 排序种群及其适应度列表
        sort_population = deepcopy(population)
        sort_population_fitnesses = [self.gene_fitness(gene) for gene in population]
        # 冒泡排序 次数与目标种群规模相同 后面部分不需要排序
        for i_gene in range(num):
            for i_gene1 in range(i_gene+1, len(sort_population)):
                # 前面任务低于后面任务则交换
                if sort_population_fitnesses[i_gene] < sort_population_fitnesses[i_gene1]:
                    temp_gene = deepcopy(sort_population[i_gene])
                    sort_population[i_gene] = deepcopy(sort_population[i_gene1])
                    sort_population[i_gene1] = temp_gene
                    temp_fitness = deepcopy(sort_population_fitnesses[i_gene])
                    sort_population_fitnesses[i_gene] = deepcopy(sort_population_fitnesses[i_gene1])
                    sort_population_fitnesses[i_gene1] = temp_fitness
        # 输出产生种群
        return sort_population[0 : num]

    # 基因繁殖函数
    # gene: 父代基因型
    # task_poses:  任务点位置
    def gene_reproduct(self, gene:Gene) -> List[Gene]:
        # 子代序列基因型列表
        child_schedule_genes = [deepcopy(gene) for i_child in range(8)]
        for i_child in range(4):
            # 0: 序列基因型不变
            if i_child == 0:
                child_schedule_genes[i_child] = deepcopy(gene)
            # 1: 序列两点对调
            elif i_child == 1:
                variation_points = random.sample([x for x in range(self.num_task)], 2)
                if variation_points[0] > variation_points[1]:
                    temp = variation_points[1]
                    variation_points[1] = variation_points[0]
                    variation_points[0] = temp
                child_schedule_genes[i_child] = deepcopy(gene)
                child_schedule_genes[i_child].schedule_gene[variation_points[1]] \
                     = gene.schedule_gene[variation_points[0]]
                child_schedule_genes[i_child].schedule_gene[variation_points[0]] \
                     = gene.schedule_gene[variation_points[1]]
            # 2: 序列两点之间任务倒序
            elif i_child == 2:
                variation_points = random.sample([x for x in range(self.num_task)], 2)
                if variation_points[0] > variation_points[1]:
                    temp = variation_points[1]
                    variation_points[1] = variation_points[0]
                    variation_points[0] = temp
                child_schedule_genes[i_child] = deepcopy(gene)
                for i_change in range(variation_points[1] - variation_points[0] + 1):
                    child_schedule_genes[i_child].schedule_gene[variation_points[0] + i_change] \
                         = gene.schedule_gene[variation_points[1] - i_change]
            # 3: 序列两点之间任务后移一位 末位前移
            else:
                variation_points = random.sample([x for x in range(self.num_task)], 2)
                if variation_points[0] > variation_points[1]:
                    temp = variation_points[1]
                    variation_points[1] = variation_points[0]
                    variation_points[0] = temp
                child_schedule_genes[i_child] = deepcopy(gene)
                child_schedule_genes[i_child].schedule_gene[variation_points[0]] \
                    = gene.schedule_gene[variation_points[1]]
                child_schedule_genes[i_child].schedule_gene[variation_points[0]+1 : variation_points[1]+1] \
                     = gene.schedule_gene[variation_points[0] : variation_points[1]]
        
        # 子代基因型列表
        child_genes = [deepcopy(gene) for i_child in range(8)]
        for i_child in range(4):
            # 序列基因型 0~3 4~7分别与4种序列基因型对应
            child_genes[i_child].schedule_gene = deepcopy(child_schedule_genes[i_child].schedule_gene)
            child_genes[i_child + 4].schedule_gene = deepcopy(child_schedule_genes[i_child].schedule_gene)
            # 断点基因型 0~3不变 4~7随机调整
            child_genes[i_child].breakpoint_gene = deepcopy(gene.breakpoint_gene)
            child_genes[i_child + 4].breakpoint_gene = deepcopy(gene.breakpoint_gene)
            random.shuffle(child_genes[i_child + 4].breakpoint_gene)
        return child_genes

    # 基因适应度计算函数
    # gene: 基因型
    def gene_fitness(self, gene:Gene) -> float:
        agent_schedules = self.gene_decode(gene)
        agent_rewards = [0 for i_agent in range(self.num_agent)]
        for i_agent in range(self.num_agent):
            schedule = agent_schedules[i_agent]
            if len(schedule) < 1:
                continue
            agent_time_distance = self.agent_task_dist_mat[i_agent, schedule[0]]
            agent_rewards[i_agent] = math.exp(-self.reward_time_decay * agent_time_distance)
            for i_task in range(1, len(schedule)):
                agent_time_distance += self.task_dist_mat[schedule[i_task - 1], schedule[i_task]]
                agent_rewards[i_agent] += math.exp(-self.reward_time_decay * agent_time_distance)
        total_reward = 0.0
        for i_agent in range(self.num_agent):
            total_reward += agent_rewards[i_agent]
        return total_reward

    # 基因译码函数
    # gene: 基因型
    def gene_decode(self, gene:Gene) -> List[List[int]]:
        agent_schedules = [[] for i_agent in range(self.num_agent)]
        breakpoints = [i_task for i_task, state in enumerate(gene.breakpoint_gene) if state == 1]
        breakpoints.insert(0, 0)
        breakpoints.append(len(gene.breakpoint_gene)-1)
        for i_agent in range(self.num_agent):
            agent_schedules[i_agent] = deepcopy(gene.schedule_gene[breakpoints[i_agent] : breakpoints[i_agent+1]])
        return agent_schedules

    # 任务分配距离代价评估函数
    # agent_poses: 智能体初始位置
    # task_poses:  任务点位置
    # schedules:   智能体任务序列
    # reward_decay:收益衰减系数
    def allocation_reward_eval(self, agent_poses:List[Pose2D], task_poses:List[Pose2D],\
            schedules:List[list]) -> float:
        num_agent = len(agent_poses)
        num_task = len(task_poses)
        agent_rewards = [0 for i_agent in range(num_agent)]
        for i_agent in range(num_agent):
            agent_pose = agent_poses[i_agent]
            agent_time_distance = 0
            for task in schedules[i_agent]:
                agent_time_distance += self.eval_time_estimate_func(agent_pose, task_poses[task])
                agent_rewards[i_agent] += math.exp(-self.reward_time_decay * agent_time_distance)
                agent_pose = task_poses[task]
        total_reward = 0
        for i_agent in range(num_agent):
            total_reward += agent_rewards[i_agent]
        return total_reward
    
    def allocation_distance_eval(self, agent_poses:List[Pose2D], task_poses:List[Pose2D],
                                  schedules:List[list]):
        num_agent = len(agent_poses)
        num_task = len(task_poses)
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

    # 距离计算函数
    # start_pose: 起点位置
    # goal_pose:  终点位置
    def pose_distance(self, start_pose:Pose2D, goal_pose:Pose2D) -> float:
        dx = start_pose.x - goal_pose.x
        dy = start_pose.y - goal_pose.y
        return math.sqrt(dx*dx + dy*dy)

    def dataStructTransform(self, poses:np.ndarray) -> List[Pose2D]:
        num_poses = poses.shape[0]
        pose_list = []
        for i_pose in range(num_poses):
            pose_list.append(Pose2D(x = poses[i_pose, 0], y = poses[i_pose, 1]))
        return pose_list

if __name__ == '__main__':
    # task_poses = []
    # num_task = 20
    # for i_task in range(num_task):
    #     task_poses.append(Pose2D(x = random.random(), y = random.random()))
    
    # agent_poses = []
    # num_agent = 5
    # for i_agent in range(num_agent):
    #     agent_poses.append(Pose2D(x = random.random(), y = random.random()))
    num_task = 20
    num_agent = 3
    task_poses = np.random.rand(num_task, 2)
    agent_poses = np.random.rand(num_agent, 2)
    
    planner = GATaskAllocationPlanner()
    planner.config_time_estimate_function(planner.pose_distance)
    result = planner.ga_allocate(planner.dataStructTransform(agent_poses), planner.dataStructTransform(task_poses))
    print(result)
    print(planner.allocation_distance_eval(planner.dataStructTransform(agent_poses), planner.dataStructTransform(task_poses), result))
    print(planner.allocation_reward_eval(planner.dataStructTransform(agent_poses), planner.dataStructTransform(task_poses), result))

