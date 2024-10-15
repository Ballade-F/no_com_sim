import numpy as np
from math import *
import matplotlib.pyplot as plt

cost_inf = 10000.0


#栅格地图 由格数和分辨力构成 0为可通行，1为障碍物
#轨迹，由近到远，为元组的列表（x，y）为实际坐标
#坐标系，x轴向前，y轴向左

'''
更新栅格地图

计算dwa，输入当前五维状态，输出是否有解flag，输出最优速度和角速度
    1.根据标准速度确定的前瞻和轨迹，判断是否有解以及是否要到终点
        无解则返回无解
        到终点则计算推荐速度和角度，返回推荐速度和角度，终点
        否则计算前瞻点角度，返回推荐速度和角度，前瞻点
    2.两层for遍历速度和角速度，
        计算每个速度角速度对应的轨迹，
        计算终点评价，
        计算速度评价，用推荐速度，
        计算障碍物评价，用轨迹上每个点，碰到障碍物直接否定（无穷），此处可能无解
        选取最优的速度角速度
'''
#obstacle_r 障碍势场需要考虑的范围，单位m
class DWA:
    def __init__(self, v_ave, dt, predict_time, pos_factor, theta_factor, v_factor, w_factor, obstacle_factor,
                 obstacle_r, resolution_x, resolution_y, grid_map:np.ndarray,
                 v_min = -0.4, v_max = 0.4, w_min = -0.4, w_max = 0.4, v_reso = 0.05, w_reso = 0.05):
        self.v_ave = v_ave
        self.dt = dt
        self.predict_time = predict_time
        self.pos_factor = pos_factor
        self.theta_factor = theta_factor
        self.v_factor = v_factor
        self.w_factor = w_factor
        self.obstacle_factor = obstacle_factor

        self.v_min = v_min
        self.v_max = v_max
        self.w_min = w_min
        self.w_max = w_max
        self.v_reso = v_reso
        self.w_reso = w_reso

        self.obstacle_r = obstacle_r
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        #障碍物势场需要考虑的栅格数
        self.obstacle_grid_x = int(obstacle_r/resolution_x)
        self.obstacle_grid_y = int(obstacle_r/resolution_y)

        self.n_x = grid_map.shape[0]
        self.n_y = grid_map.shape[1]
        self.grid_map = grid_map
        # #障碍物势场
        # self.filed_map = np.zeros((self.n_x,self.n_y),dtype=float)

        #实际坐标
        self.x=0.0
        self.y=0.0
        self.theta=0.0
        self.v=0.0
        self.w=0.0
        

    #输入当前状态，输出是否有解，最优速度和角速度
    def DWA_Planner(self, path, x, y, theta,v,w):
        #1.确定路径是否在范围内，得到前瞻点和推荐速度
        target_flag, target = self.getTarget(path)
        if target_flag == False:
            return False, None, None
    
        #2.遍历速度和角速度，计算评价函数
        best_u = [0.0,0.0]
        best_cost = cost_inf
        for v in np.arange(self.v_min, self.v_max, self.v_reso):
            for w in np.arange(self.w_min, self.w_max, self.w_reso):
                #计算评价函数
                collision_flag, cost = self.calculateCost([x,y,theta,v,w], [v,w], target)
                if collision_flag == True:
                    continue
                if cost < best_cost:
                    best_cost = cost
                    best_u = [v,w]
        if best_cost > cost_inf/2:
            target_flag = False
        return target_flag, best_u[0], best_u[1]
    
    #计算评价函数
    #state = [x,y,theta,v,w]
    #u = [v,w]
    #target = [x,y,theta,v_ref]
    def calculateCost(self, state, u, target):
        #计算轨迹
        traj, final_state = self.calculate_traj(state, u)
        #计算评价函数
        goal_cost = self.goal_cost([target[0],target[1],target[2]],[final_state[0],final_state[1],final_state[2]])
        velocity_cost = self.u_cost(u,[target[3],0])
        collision_flag, obstacle_cost = self.obstacle_cost(traj)
        cost = goal_cost + velocity_cost + obstacle_cost
        return collision_flag, cost

    def Update_GridMap(self, grid_map:np.ndarray):
        self.grid_map = grid_map
        self.n_x = grid_map.shape[0]
        self.n_y = grid_map.shape[1]
        # #障碍物势场,用引力函数表代价
        # for i in range(self.n_x):
        #     for j in range(self.n_y):
        #         if self.grid_map[i][j] == 1:
        #             self.filed_map[i][j] = 10000.0
        #         else:
        #             self.filed_map[i][j] = 1.0/self.obstacle_cost([[i*self.resolution_x,j*self.resolution_y]])
        return True

    #返回是否有解，[x,y,theta,v_ref]
    def getTarget(self, path):
        #确定是否有解以及是否到终点
        target_flag = False
        finish_flag = True
        find_r = self.v_ave * self.predict_time
        target = np.zeros((4),dtype=float)
        target_index = 0
        for i ,point in enumerate(path):
            if sqrt((point[0]-self.x)**2+(point[1]-self.y)**2) < find_r:
                target_flag = True
            elif target_flag == True:
                target_index = i
                finish_flag = False
                break
        
        #离太远
        if target_flag == False:
            return target_flag, target
        #到终点
        if finish_flag == True:
            t_point = path[-1]
            t_last_point = path[-2]
            #弧度
            theta_ref = atan2(t_point[1]-t_last_point[1],t_point[0]-t_last_point[0])
            t_r = sqrt((t_point[0]-self.x)**2+(t_point[1]-self.y)**2)
            v_ref = t_r/find_r * self.v_ave
            target[0] = t_point[0]
            target[1] = t_point[1]
            target[2] = theta_ref
            target[3] = v_ref
        else:
            t_point = path[target_index]
            t_last_point = path[target_index-1]
            #弧度
            theta_ref = atan2(t_point[1]-t_last_point[1],t_point[0]-t_last_point[0])
            target[0] = t_point[0]
            target[1] = t_point[1]
            target[2] = theta_ref
            target[3] = self.v_ave
        return target_flag, target

            
    def forward(self, state, u):
        state[0] += u[0]*self.dt*cos(state[2])
        state[1] += u[0]*self.dt*sin(state[2])
        state[2] += u[1]*self.dt
        state[3] = u[0]
        state[4] = u[1]
        return state
    
    #给定初状态和速度角速度，计算轨迹和终点状态
    def calculate_traj(self, state, u):
        traj_len = int(self.predict_time / self.dt)
        traj = [None for _ in range(traj_len)]
        for i in range(traj_len):
            state = self.forward(state, u)
            traj[i] = (state[0],state[1])
        return traj, state

    #u = [v,w] 采样选择的速度和角速度
    #u_ref = [v_ref,w_ref] 参考u，可以取当前角速度、参考速度
    def u_cost(self, u, u_ref):
        cost = sqrt((self.v_factor*(u[0]-u_ref[0]))**2+(self.w_factor*(u[1]-u_ref[1]))**2)
        return cost
    
    #final_state预测轨迹终点[x,y,theta]
    #goal目标点[x,y,theta]
    def goal_cost(self, goal, final_state):
        cost = sqrt(self.pos_factor*(final_state[0]-goal[0])**2+self.pos_factor*(final_state[1]-goal[1])**2+self.theta_factor*(final_state[2]-goal[2])**2)
        return cost
    
    #traj 为预测轨迹，（x，y）列表； filed_flag是否考虑势场
    #返回是否碰撞，以及cost，这里的cost进行归一化
    def obstacle_cost(self, traj, filed_flag = False):
        n_point = len(traj)
        collision_flag = False
        cost = 0.0
        for i,point in enumerate(traj):
            x = int(point[0]/self.resolution_x)
            y = int(point[1]/self.resolution_y)
            if x < 0 or x >= self.n_x or y < 0 or y >= self.n_y:
                collision_flag = True
            if self.grid_map[x][y] == 1:
                collision_flag = True
            if filed_flag == True:
                #势场法
                ob_counter = 0
                cost_point = 0.0
                for j in range(-self.obstacle_grid_x,self.obstacle_grid_x+1):
                    for k in range(-self.obstacle_grid_y,self.obstacle_grid_y+1):
                        if x+j < 0 or x+j >= self.n_x or y+k < 0 or y+k >= self.n_y or self.grid_map[x+j][y+k] == 1:
                            ob_counter += 1
                            cost_point += 1.0/(j**2+k**2)
                if ob_counter > 0:
                    cost_point /= ob_counter
                cost += cost_point
        cost /= n_point
        return collision_flag, cost
            


        
        
        

    

    