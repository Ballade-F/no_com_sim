import numpy as np
from math import *
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    def __init__(self, v_ave, dt, predict_time, pos_factor, theta_factor, v_factor, w_factor, obstacle_factor, final_factor,
                 obstacle_r, resolution_x, resolution_y, grid_map:np.ndarray,ob_filed_flag = False,
                 v_min = -0.3, v_max = 0.3, w_min = -0.3, w_max = 0.3, v_reso = 0.05, w_reso = 0.05, n_workers = 4):
        self.v_ave = v_ave
        self.dt = dt
        self.predict_time = predict_time
        self.pred_len = int(predict_time/dt)
        self.pos_factor = pos_factor
        self.theta_factor = theta_factor
        self.v_factor = v_factor
        self.w_factor = w_factor
        self.obstacle_factor = obstacle_factor
        self.final_factor = final_factor

        self.v_min = v_min
        self.v_max = v_max
        self.w_min = w_min
        self.w_max = w_max
        self.v_reso = v_reso
        self.w_reso = w_reso

        self.n_workers = n_workers

        self.obstacle_r = obstacle_r
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        #障碍物势场需要考虑的栅格数
        self.obstacle_grid_x = int(obstacle_r/resolution_x)
        self.obstacle_grid_y = int(obstacle_r/resolution_y)
        self.ob_filed_flag = ob_filed_flag

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

        self.finish_flag = False
        

    #输入当前状态，输出是否有解，最优速度和角速度
    def DWA_Planner(self, path, state_0):
        #1.确定路径是否在范围内，得到前瞻点和推荐速度
        w_ref = state_0[4]
        self.x = state_0[0]
        self.y = state_0[1]
        self.theta = state_0[2]
        self.v = state_0[3]
        self.w = state_0[4]
        target_flag, out_path, v_ref, _ = self.getTarget(path)
        if target_flag == False:
            return False, None, None
    
        #2.遍历速度和角速度，计算评价函数
        best_u = [0.0,0.0]
        best_cost = cost_inf
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for v in np.arange(self.v_min, self.v_max, self.v_reso):
                for w in np.arange(self.w_min, self.w_max, self.w_reso):
                    #计算评价函数
                    state = state_0.copy()
                    u = [v,w].copy()
                    futures.append(executor.submit(self.calculateCost, state, u, out_path, v_ref, w_ref))
            for future in futures:
                collision_flag, cost, u_out = future.result()
                if collision_flag == False and cost < best_cost:
                    best_cost = cost
                    best_u = u_out
        # for v in np.arange(self.v_min, self.v_max, self.v_reso):
        #     for w in np.arange(self.w_min, self.w_max, self.w_reso):
        #         #计算评价函数
        #         state = state_0.copy()
        #         collision_flag, cost = self.calculateCost(state, [v,w], out_path, v_ref, w_ref)
        #         if collision_flag == True:
        #             continue
        #         if cost < best_cost:
        #             best_cost = cost
        #             best_u = [v,w]
        if best_cost > cost_inf/2:
            target_flag = False
        return target_flag, best_u[0], best_u[1]
    
    #计算评价函数
    #state = [x,y,theta,v,w]
    #u = [v,w]
    #ref_path = [x,y,theta]的np数组，shape为[n,3]
    def calculateCost(self, state, u, ref_path, ref_v, ref_w):
        #计算轨迹
        traj, final_state = self.calculate_traj(state, u)
        #计算评价函数
        traj_cost = self.traj_cost(ref_path, traj)
        velocity_cost = self.u_cost(u,[ref_v,ref_w])
        collision_flag, obstacle_cost = self.obstacle_cost(traj,self.ob_filed_flag)
        cost = traj_cost + velocity_cost + obstacle_cost
        return collision_flag, cost, u

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

    #返回是否有解，[x,y,theta]的np数组，shape为[n,3], v_ref
    def getTarget(self, path):
        #确定是否有解以及是否到终点
        target_flag = False
        finish_flag = True
        find_r = self.v_max * self.predict_time
        target_path = []
        v_ref = self.v_ave
        for i ,point in enumerate(path):
            if sqrt((point[0]-self.x)**2+(point[1]-self.y)**2) < find_r:
                if not target_flag:
                    # begin_idx = i
                    target_flag = True
                target_path.append(point)
            elif target_flag == True:
                # target_index = i
                finish_flag = False
                break
        
        #离太远
        if target_flag == False:
            return target_flag, None, None, None
        #到终点
        self.finish_flag = finish_flag
        # if finish_flag == True:
        #     t_point = path[-1]
        #     t_r = sqrt((t_point[0]-self.x)**2+(t_point[1]-self.y)**2)
        #     v_ref = t_r/find_r * self.v_ave


        #计算theta
        n = len(target_path)
        out_path = np.zeros((n,3),dtype=float)
        for i in range(n):
            if i == n-1:
                out_path[i][0] = target_path[i][0]
                out_path[i][1] = target_path[i][1]
                out_path[i][2] = atan2(target_path[i][1]-target_path[i-1][1],target_path[i][0]-target_path[i-1][0])
            else:
                out_path[i][0] = target_path[i][0]
                out_path[i][1] = target_path[i][1]
                #弧度
                out_path[i][2] = atan2(target_path[i+1][1]-target_path[i][1],target_path[i+1][0]-target_path[i][0])

        #计算w_ref
        t_point = path[-1]
        theta_ref = atan2(t_point[1]-self.y,t_point[0]-self.x)
        w_ref = (theta_ref-self.theta)/self.predict_time
        w_ref = max(min(w_ref,self.w_max),self.w_min)

        return target_flag, out_path, v_ref, w_ref

            
    def forward(self, state, u):
        state[0] += u[0]*self.dt*cos(state[2])
        state[1] += u[0]*self.dt*sin(state[2])
        state[2] += u[1]*self.dt
        state[3] = u[0]
        state[4] = u[1]
        return state
    
    #给定初状态和速度角速度，计算轨迹和终点状态
    def calculate_traj(self, state, u):
        traj = np.zeros((self.pred_len,3),dtype=float)
        for i in range(self.pred_len):
            state = self.forward(state, u)
            traj[i,:] = state[0:3]
        return traj, state

    #u = [v,w] 采样选择的速度和角速度
    #u_ref = [v_ref,w_ref] 参考u，可以取当前角速度、参考速度
    def u_cost(self, u, u_ref):
        cost = self.v_factor*(u[0]-u_ref[0])**2+self.w_factor*(u[1]-u_ref[1])**2
        return cost
    
    #pred_path [x,y,theta]的np数组，shape为[self.pred_len,3]
    #ref_path [x,y,theta]的np数组，shape为[n,3]
    def traj_cost(self, ref_path, pred_path):
        pos_cost = 0.0
        theta_cost = 0.0
        for i in range(self.pred_len):
            dx = pred_path[i,0] - ref_path[:,0]
            dy = pred_path[i,1] - ref_path[:,1]
            dist = dx**2+dy**2
            min_idx = np.argmin(dist)
            pos_cost += dist[min_idx]
            theta_cost += (pred_path[i,2]-ref_path[min_idx,2])**2
        pos_cost /= self.pred_len
        theta_cost /= self.pred_len
        final_pos_cost = self.final_factor*(pred_path[-1,0]-ref_path[-1,0])**2 + (pred_path[-1,1]-ref_path[-1,1])**2
        final_theta_cost = self.final_factor*(pred_path[-1,2]-ref_path[-1,2])**2
        cost = self.pos_factor*(pos_cost+final_pos_cost) + self.theta_factor*(theta_cost+final_theta_cost)
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
            elif self.grid_map[x][y] == 1:
                collision_flag = True
            if filed_flag == True and collision_flag == False:
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
        cost = cost/n_point * self.obstacle_factor
        return collision_flag, cost
            


        
        
        

    

    