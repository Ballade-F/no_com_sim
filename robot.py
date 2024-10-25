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
    def __init__(self, map:mp.Map, cfg_dir:str, device='cpu') -> None:
    # 成员变量
        # 基本信息
        self.robot_id = robot_id
        self.static_map = deepcopy(map)
        self.map = deepcopy(map)
        self.robot_r = 0.2
        self.robot_r_idx = round(self.robot_r / map.resolution_x)
        self.device = device
        self.n_robot = n_robot
        self.n_task = n_task

        self.n_ob_points = n_ob_points
        self.n_obstacle = n_obstacle

        # 感知决策
        
        self.buffer_size = buffer_size
        self.buffer_robots = ringbuffer.RingBuffer(self.buffer_size) # 其中元素类型为RobotData
        self.buffer_tasks = ringbuffer.RingBuffer(self.buffer_size) # 其中元素类型为TaskData
        self.buffer_decisions = ringbuffer.RingBuffer(self.buffer_size)
        self.target = []  # n_robot个元素，每个元素是一个任务的index

        # 归一化坐标
        self.feature_obstacle = torch.empty(self.n_obstacle, self.n_ob_points, 2, dtype=torch.float32, device=self.device)
        for i_ob in range(self.n_obstacle):
            self.feature_obstacle[i_ob] = self.static_map.obstacles[i_ob]
        self.feature_obstacle = self.feature_obstacle.unsqueeze(0)# (1, n_obstacle, n_ob_points, 2)

        self.robot_data = RobotData(self.n_robot)
        self.task_data = TaskData(self.n_task)
        self.path = []
        self.task_list = []
        self.robot_intention = np.full((self.n_robot), -1, dtype=int)
        
        # 控制
        self.x = 0
        self.y = 0
        self.theta = 0
        self.w = 0
        self.v = 0

        # 调度用的标志量
        self.base_T = 0.1  # 最小调度周期，控制与感知周期
        self.keyframe_counter = 5  # 关键帧周期数
        # self.decision_counter = 25  # 决策周期数

        self.counter_base = 0

        self.replan_flag = False
        self.reallocation_flag = False
        self.stop_flag = False


    # 算法类实例与网络类实例
        dwa_planner = dwa.DWA(v_ave, dt, predict_time, pos_factor, theta_factor, v_factor, w_factor, obstacle_factor,final_factor,
                           obstacle_r, self.map.resolution_x, self.map.resolution_y, self.map.grid_map,True,n_workers=4)
        self.path_planner = path_planner.AStarPlanner(self.map.grid_map, self.map.resolution_x, self.map.resolution_y)
        #TODO: 换成网络
        self.task_allocation = greedy.GreedyTaskAllocationPlanner()
        self.intention_judgment = intention.IntentionNet()
        self.intention_judgment.config(intention_config)
        self.intention_judgment.load_state_dict(torch.load(intention_model_dir))



    # 用感知信息更新状态
    def updateState(self, robot_data, task_data):
        # 更新机器人占用的地图
        for i in range(self.n_robot):
            x_idx = int(self.robot_data[i][0]/self.map.resolution_x)
            y_idx = int(self.robot_data[i][1]/self.map.resolution_y)
            for j in range(-self.robot_r_idx, self.robot_r_idx+1):
                for k in range(-self.robot_r_idx, self.robot_r_idx+1):
                    grid_x = max(0, min(x_idx+j, self.map.n_x-1))
                    grid_y = max(0, min(y_idx+k, self.map.n_y-1))
                    if self.static_map.grid_map[grid_x][grid_y] == 0:
                        self.map.grid_map[grid_x][grid_y] = 0
        self.robot_data = robot_data
        for i in range(self.n_robot):
            if i == self.robot_id:
                continue
            x_idx = int(self.robot_data[i][0]/self.map.resolution_x)
            y_idx = int(self.robot_data[i][1]/self.map.resolution_y)
            for j in range(-self.robot_r_idx, self.robot_r_idx+1):
                for k in range(-self.robot_r_idx, self.robot_r_idx+1):
                    grid_x = max(0, min(x_idx+j, self.map.n_x-1))
                    grid_y = max(0, min(y_idx+k, self.map.n_y-1))
                    self.map.grid_map[grid_x][grid_y] = 1

        # 更新自身状态
        self.x = self.robot_data[self.robot_id][0]
        self.y = self.robot_data[self.robot_id][1]
        self.theta = self.robot_data[self.robot_id][2]

        # 更新任务状态
        self.task_data = task_data
        while len(self.task_list) > 0:
            if self.task_data[self.task_list[0]][2] == 1:
                self.task_list.pop(0)
                self.replan_flag = True
            else:
                break
        if len(self.task_list) == 0:
            self.stop_flag = True
            self.replan_flag = False

    def updatePlan(self):
        if self.replan_flag:
            _start = self.map.true2grid([self.x, self.y])
            _task = self.map.true2grid([self.task_data[self.task_list[0]][0], self.task_data[self.task_list[0]][1]])
            self.path,_cost = self.path_planner.plan(_start, _task, reset_nodes=True, grid_mode=False)
            self.replan_flag = False
            self.stop_flag = False
            if len(self.path) == 1:
                self.stop_flag = True
                    
    def updataControl(self):
        if self.stop_flag:
            self.v = 0
            self.w = 0
            return (self.v, self.w)
        
        success_flag, self.v, self.w = self.dwa_planner.DWA_Planner(self.path, [self.x, self.y, self.theta, self.v, self.w])
        if not success_flag:
            self.replan_flag = True
            self.v = 0
            self.w = 0
        return (self.v, self.w)
    
    def updateKeyframe(self):
        self.buffer_robots.push(self.robot_data)
        self.buffer_tasks.push(self.task_data)

    def updateIntention(self):
        robot_traj = torch.empty(self.n_robot, self.buffer_size, 2,dtype=torch.float32, device=self.device)
        task_traj = torch.empty(self.n_task, self.buffer_size, 3,dtype=torch.float32, device=self.device)
        for i in range(self.buffer_size):
            # intention网络里面 0号是最老的，buffer里面0号是最新的
            robot_traj[:,i,:] = self.buffer_robots[self.buffer_size-i-1].data[:,:2]
            task_traj[:,i,:] = self.buffer_tasks[self.buffer_size-i-1].data[:,:]
        robot_traj = robot_traj.unsqueeze(0)# (1, n_robot, buffer_size, 2)
        task_traj = task_traj.unsqueeze(0)# (1, n_task, buffer_size, 3)
        intentionProba_rt = self.intention_judgment(robot_traj, task_traj, self.feature_obstacle,is_train=False)# (1, n_robot, n_task)
        # 选择概率最大的任务，如果没有任务，选择n_task
        for i in range(self.n_robot):
            self.robot_intention[i] = torch.argmax(intentionProba_rt[0,i]).item()


        
    
    def updateDecision(self):
        pass
    

    def control_callback(self):
        self.updateState()
        self.updatePlan()
        self.updataControl()

    def keyframe_callback(self):
        self.updateState()
        self.updateKeyframe()
        self.updateIntention()
        self.updateDecision()
        self.updatePlan()
        self.updataControl()






# 最小周期回调函数
    def base_callback(self):
        self.counter_base += 1
        if self.counter_base % self.keyframe_counter == 0:
            self.keyframe_callback()
        else :
            self.control_callback()
