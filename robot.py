import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import json

import utils.map as mp
import utils.ringbuffer as ringbuffer
import pnc.dwa as dwa
import pnc.path_planner as path_planner
import task_allocation.hungarian as hungarian
import task_allocation.greedy_allocation_lib as greedy
import task_allocation.ga_allocation_lib as ga
import net.allocation as allocation
import net.intention as intention


class Robot:
    def __init__(self, map:mp.Map, cfg_dir:str,robot_init_state,task_init_state) -> None:
    # 成员变量
    #重新整理
    # 基本信息
        with open(cfg_dir, "r") as f:
            cfg = json.load(f)
        self.robot_id = cfg["robot_id"]
        self.robot_r = cfg["robot_r"]  # 机器人占用半径
        self.device = cfg["device"]
        self.static_map = deepcopy(map) # 静态地图，A*用
        self.map = deepcopy(map)        # 动态地图，DWA用
        self.robot_r_idx = round(self.robot_r / map.resolution_x) # 机器人半径对应的网格数
        self.n_ob_points = map.n_ob_points # 障碍物多边形边数
        self.n_robot = map.n_starts
        self.n_task = map.n_tasks
        self.n_rt = self.n_robot + self.n_task
        self.n_obstacle = map.n_obstacles
    # 外部输入，实际坐标    
        # self.robot_data = np.zeros((self.n_robot, 3))# x, y, theta
        # self.task_data = np.zeros((self.n_task, 3))# x, y, if_finished
        self.robot_data = robot_init_state.copy()
        self.task_data = task_init_state.copy()
    # 控制,x,y,theta从感知获取，v,w从控制获取,都是实际坐标，角度为弧度，速度为m/s
        self.x = self.robot_data[self.robot_id][0]
        self.y = self.robot_data[self.robot_id][1]
        self.theta = self.robot_data[self.robot_id][2]
        self.w = 0
        self.v = 0
        v_ave = cfg["v_ave"]
        dt = cfg["dt"]
        predict_time = cfg["predict_time"]
        pos_factor = cfg["pos_factor"]
        theta_factor = cfg["theta_factor"]
        v_factor = cfg["v_factor"]
        w_factor = cfg["w_factor"]
        obstacle_factor = cfg["obstacle_factor"]
        final_factor = cfg["final_factor"]
        obstacle_r = cfg["obstacle_r"]
        self.dwa_planner = dwa.DWA(v_ave, dt, predict_time, pos_factor, theta_factor, v_factor, w_factor, obstacle_factor,final_factor,
                           obstacle_r, self.map.resolution_x, self.map.resolution_y, self.map.grid_map,True,n_workers=4)
    # 规划，实际坐标
        self.path = []
        self.path_planner = path_planner.AStarPlanner(self.static_map.grid_map, self.static_map.resolution_x, self.static_map.resolution_y)
        
    # 意图判断
        self.buffer_size = cfg["buffer_size"]
        self.buffer_robots = ringbuffer.RingBuffer(self.buffer_size) # 其中元素类型为np.ndarray((n_robot,3))
        self.buffer_tasks = ringbuffer.RingBuffer(self.buffer_size) # 其中元素类型为np.ndarray((n_robot,3))
        # 归一化坐标,x_ob: (batch, n_obstacle, ob_points, 2)
        self.feature_obstacle = torch.empty(1, self.n_obstacle, self.n_ob_points, 2, dtype=torch.float32, device=self.device)
        for i_ob in range(self.n_obstacle):
            self.feature_obstacle[0,i_ob] = torch.from_numpy(self.static_map.obstacles[i_ob])

        self.robot_intention = np.full((self.n_robot), -1, dtype=int)

        self.intention_judgment = intention.IntentionNet()
        intention_model_dir = cfg["intention_model_dir"]
        self.intention_judgment.load_state_dict(torch.load(intention_model_dir, map_location=self.device))
        self.intention_judgment.config({'n_robot':self.n_robot, 
                                        'n_task':self.n_task, 
                                        'n_obstacle':self.n_obstacle})
        
    # 任务分配
        self.task_list = []# 任务序号
        #TODO: 换成网络
        self.task_allocation = greedy.GreedyTaskAllocationPlanner()

    # 调度用的标志量
        self.base_T = 0.1  # 最小调度周期，控制与感知周期
        self.keyframe_counter = 5  # 关键帧周期数
        # self.decision_counter = 25  # 决策周期数
        self.counter_base = 0
        self.replan_flag = False
        self.stop_flag = False

    # 初始化
        self.updateKeyframe()
        # 下三角矩阵
        self.costmat = np.full((self.n_task, self.n_task),fill_value=-1.0,dtype=float)
        for j in range(self.n_task):
            for k in range(j+1):
                # astar_planner.resetNodes()
                self.costmat[j, k] = self.path_planner.plan(map.tasks_grid[j], map.tasks_grid[k],path_flag=False)
                # 对称矩阵
                self.costmat[k, j] = self.costmat[j, k]
        # 初始化任务列表
        self._reallocation()
    
    

# 回调函数
    def base_callback(self,robot_data, task_data):
        '''
        最小周期回调函数    
        robot_data: ndarray((n_robot,5)), 真实坐标  
        task_data: ndarray((n_task,3))，真实坐标   
        return: (v, w) 
        '''
        self.counter_base += 1
        out = None
        # if self.counter_base % self.keyframe_counter == 0:
        #     out = self.keyframe_callback(robot_data, task_data)
        # else :
        #     out = self.control_callback(robot_data, task_data)
        out = self.control_callback(robot_data, task_data)
        return out
    
    # output: (v, w)
    def control_callback(self,robot_data, task_data):
        self.updateState(robot_data, task_data)
        self.updatePlan()
        out = self.updataControl()
        return out
    
    # output: (v, w)
    def keyframe_callback(self,robot_data, task_data):
        self.updateState(robot_data, task_data)
        self.updateKeyframe()
        if len(self.buffer_robots) == self.buffer_size:
            self.updateIntention()
        self.updateDecision()
        self.updatePlan()
        out = self.updataControl()
        return out



    # 用感知信息更新状态
    def updateState(self, robot_data, task_data):
        '''
        robot_data: ndarray((n_robot,3)), 真实坐标  
        task_data: ndarray((n_task,3))，真实坐标    
        '''
        # 更新机器人占用的地图
        # 恢复上一次占用
        for i in range(self.n_robot):
            x_idx = int(self.robot_data[i][0]/self.map.resolution_x)
            y_idx = int(self.robot_data[i][1]/self.map.resolution_y)
            for j in range(-self.robot_r_idx, self.robot_r_idx+1):
                for k in range(-self.robot_r_idx, self.robot_r_idx+1):
                    grid_x = max(0, min(x_idx+j, self.map.n_x-1))
                    grid_y = max(0, min(y_idx+k, self.map.n_y-1))
                    if self.static_map.grid_map[grid_x][grid_y] == 0:
                        self.map.grid_map[grid_x][grid_y] = 0
        # 更新当前占用
        self.robot_data = robot_data.copy()
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
        self.task_data = task_data.copy()
        # 只关注最近一个任务，任务点改变时，重新规划
        while len(self.task_list) > 0:
            if self.task_data[self.task_list[0]][2] == 1:
                self.task_list.pop(0)
                self.replan_flag = True
            else:
                break
        # 任务列表为空，停止
        if len(self.task_list) == 0:
            self.stop_flag = True
            self.replan_flag = False

    # A*规划路径
    def updatePlan(self):
        if self.replan_flag:
            _start = [self.x, self.y]
            _task = [self.task_data[self.task_list[0]][0], self.task_data[self.task_list[0]][1]]
            self.path,_cost = self.path_planner.plan(_start, _task, reset_nodes=True, grid_mode=False) #输入实际坐标，输出实际坐标
            self.replan_flag = False
            # 无法到达
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
        #x_r: (batch, n_robot,r_points, 2), x_t: (batch, self.n_task, 3), x_ob: (batch, n_obstacle, ob_points, 2)
        robot_traj = torch.empty(1, self.n_robot, self.buffer_size, 2,dtype=torch.float32, device=self.device)
        for i in range(self.buffer_size):
            # intention网络里面 0号是最老的，buffer里面0号是最新的，并归一化
            robot_traj[0,:,i,0] = torch.from_numpy(self.buffer_robots[i][:,0]) / (self.map.n_x*self.map.resolution_x)
            robot_traj[0,:,i,1] = torch.from_numpy(self.buffer_robots[i][:,1]) / (self.map.n_y*self.map.resolution_y)
        #将第2维度翻转
        robot_traj = torch.flip(robot_traj, [2]) 

        #最后一个task是虚拟task，用于结束任务的robot
        task_traj = torch.empty(1, self.n_task+1, 3,dtype=torch.float32, device=self.device)
        task_traj[0,-1,0] = -1
        task_traj[0,-1,1] = -1
        task_traj[0,-1,2] = 0
        task_traj[0,:-1,2] = torch.from_numpy(self.buffer_tasks[0][:,2])
        task_traj[0,:-1,0] = torch.from_numpy(self.buffer_tasks[0][:,0]) / (self.map.n_x*self.map.resolution_x)
        task_traj[0,:-1,1] = torch.from_numpy(self.buffer_tasks[0][:,1]) / (self.map.n_y*self.map.resolution_y)
        
        intentionProba_rt = self.intention_judgment(robot_traj, task_traj, self.feature_obstacle,is_train=False)# (1, n_robot, n_task)
        # 选择概率最大的任务，如果没有任务，选择n_task
        for i in range(self.n_robot):
            self.robot_intention[i] = torch.argmax(intentionProba_rt[0,i]).item()


        
    
    def updateDecision(self):
        #重分配标志
        reallocation_flag = False
        #列表为空
        if len(self.task_list) == 0:
            reallocation_flag = True
        #任务列表中有任务已经被完成
        for i in range(len(self.task_list)):
            if self.task_data[self.task_list[i]][2] == 1:
                reallocation_flag = True
                break

        cost_rt = None
        cost_t = None
        taskidx_unfinished = None

        if reallocation_flag:
            #重分配
            cost_rt, cost_t, taskidx_unfinished = self._reallocation()
            if len(self.task_list) == 0:
                self.stop_flag = True
                self.replan_flag = False
                return
        #冲突消解
        
        while True:
            conflict_flag = False
            task_un_id=-1
            robot_conflict_id=-1
            if len(self.task_list)==0 :
                self.stop_flag = True
                self.replan_flag = False
                return
            #检查是否有其他机器人的任务比自己的任务更快
            for i in range(self.n_robot):
                if i == self.robot_id:
                    continue
                #意图相同
                if self.task_list[0] == self.robot_intention[i] :
                    task_id = self.task_list[0]
                    cost_other = self.path_planner.plan(self.robot_data[i][:2], self.task_data[task_id][:2],
                                                        grid_mode=False,path_flag=False)
                    cost_self = self.path_planner.plan(self.robot_data[self.robot_id][:2], self.task_data[task_id][:2],
                                                        grid_mode=False,path_flag=False)
                    #代价还小于自己的代价,冲突
                    if cost_other < cost_self:
                        conflict_flag = True
                        robot_conflict_id = i
                        #找到该task在未完成任务中的序号，用于修改cost_rt
                        task_un_id = taskidx_unfinished.index(task_id)
                        break
            #如果没有冲突，退出
            if not conflict_flag:
                break
            #如果有冲突，重新分配;如果已经重分配过，修改矩阵
            elif reallocation_flag:
                #让出任务
                cost_rt[robot_conflict_id,task_un_id] = 0
                #重新分配
                task_temp = self.task_allocation.allocate(cost_rt, cost_t)[self.robot_id]
                self.task_list =[taskidx_unfinished[idx] for idx in task_temp]
            #如果没有重分配过，计算矩阵
            else:
                cost_rt, cost_t, taskidx_unfinished = self._reallocation()
                reallocation_flag = True
        
        #没有冲突，直接执行
        self.stop_flag = False
        if reallocation_flag:
            self.replan_flag = True
                
    def _reallocation(self):
        #选择出还未被完成的任务序号
        _taskidx_unfinished = []
        for i in range(self.n_task):
            if self.task_data[i][2] == 0:
                _taskidx_unfinished.append(i)
        n_task_unfinished = len(_taskidx_unfinished)

        #构建cost_rt
        cost_rt = np.zeros((self.n_robot, n_task_unfinished))
        for i in range(self.n_robot):
            for j in range(n_task_unfinished):
                cost_rt[i,j] = self.path_planner.plan(self.robot_data[i][:2], self.task_data[_taskidx_unfinished[j]][:2],
                                                        grid_mode=False,path_flag=False)
        #构建cost_t
        cost_t = np.zeros((n_task_unfinished, n_task_unfinished))
        for i in range(n_task_unfinished):
            for j in range(i+1):
                cost_t[i,j] = self.costmat[_taskidx_unfinished[i],_taskidx_unfinished[j]]
                cost_t[j,i] = cost_t[i,j]
        
        #分配
        # self.task_list = self.task_allocation.allocate(cost_rt, cost_t)[self.robot_id]
        task_temp = self.task_allocation.greedy_allocate_mat(cost_rt, cost_t)[self.robot_id]
        self.task_list =[_taskidx_unfinished[idx] for idx in task_temp]

        return cost_rt, cost_t, _taskidx_unfinished
        
    








