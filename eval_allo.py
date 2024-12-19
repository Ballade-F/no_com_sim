import sys
sys.path.append('./net')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import json
import csv
import os
import time

import utils.map as mp
import utils.ringbuffer as ringbuffer
import pnc.dwa as dwa
import pnc.path_planner as path_planner
import task_allocation.hungarian as hungarian
import task_allocation.greedy_allocation_lib as greedy
import task_allocation.ga_allocation_lib as ga
import net.allocation as allocation
from concurrent.futures import ProcessPoolExecutor

class evalData():
    def __init__(self, dir, norm_factor):
        batch_info = os.path.join(dir, "batch_info.json")
        with open(batch_info, "r") as f:
            self.batch_info = json.load(f)
        self.n_robot = self.batch_info["n_robot"]
        self.n_task = self.batch_info["n_task"]
        self.n_obstacle = self.batch_info["n_obstacle"]
        # self.batch_size = self.batch_info["batch_size"]
        self.batch_size = 100
        self.ob_points = self.batch_info["ob_points"]

        self.cfg = {
            "n_robot": int(self.n_robot),
            "n_task": int(self.n_task),
            "n_obstacle": int(self.n_obstacle),
            "ob_points": int(self.ob_points),
            # "batch_size": int(self.batch_size)
            "batch_size": 1
        }

        self.norm_factor = norm_factor

        self.matrix_time = -1

        self.feature_robot = np.full((self.batch_size, self.n_robot, 3),-1, dtype=float)
        self.feature_task = np.full((self.batch_size, self.n_task, 3),-1, dtype=float) 
        self.feature_obstacle = np.full((self.batch_size, self.n_obstacle, self.ob_points, 4),-1, dtype=float)
        self.costmats = np.full((self.batch_size, self.n_robot+self.n_task, self.n_robot+self.n_task),-1, dtype=float)

        for i in range(self.batch_size):
            map_dir = os.path.join(dir, f"map_{i}")
            costmat = np.load(os.path.join(map_dir, "costmat.npy"))
            self.costmats[i,:,:] = costmat * self.norm_factor
            self.costmats[i,:,:self.n_robot] = 0
            # debug, 检验costmat里是否有inf
            if np.isinf(costmat).any():
                print(f"{dir} costmat {i} has inf")
            with open(os.path.join(map_dir, "info.csv"), "r") as f:
                reader = csv.reader(f)
                for idx, row in enumerate(reader):
                    #表头
                    if idx == 0:
                        continue
                    if idx <= self.n_robot:
                        self.feature_robot[i,idx-1,:2] = row[1:]
                        self.feature_robot[i,idx-1,2] = row[0]
                    elif idx <= self.n_robot+self.n_task:
                        self.feature_task[i,idx-self.n_robot-1,:2] = row[1:]
                        self.feature_task[i,idx-self.n_robot-1,2] = row[0]
                    else:
                        idx_ob = int(row[0])-1
                        idx_point = (idx-self.n_robot-self.n_task-1)-idx_ob*self.ob_points
                        self.feature_obstacle[i, idx_ob, idx_point,0:2] = row[1:]

            #每个obs的各个点收尾相接
            for j in range(self.ob_points-1):
                self.feature_obstacle[i,:,j,2:] = self.feature_obstacle[i,:,j+1,:2]
            self.feature_obstacle[i,:,-1,2:] = self.feature_obstacle[i,:,0,:2]

    def calcu_time(self, n_x,n_y,resolution_x,resolution_y,map_idx=0):
        map = mp.Map(self.n_obstacle,self.n_robot,self.n_task,n_x,n_y,resolution_x,resolution_y,self.ob_points)
        map.setMapNorm(self.feature_obstacle[map_idx,:,:,0:2],self.feature_robot[map_idx,:,:2],self.feature_task[map_idx,:,:2])
        astar_planner = path_planner.AStarPlanner(map.grid_map, map.resolution_x, map.resolution_y)
        starts = map.starts_grid
        tasks = map.tasks_grid
        points = np.concatenate((starts, tasks), axis=0)
        start_time = time.time()
        # 下三角矩阵
        # costmat = np.full((n_robot+n_task, n_robot+n_task),fill_value=-1.0,dtype=float)
        for j in range(self.n_robot+self.n_task):
            for k in range(j+1):
                # astar_planner.resetNodes()
                cost = astar_planner.plan(points[j], points[k],path_flag=False)
                # 对称矩阵
                # costmat[k, j] = costmat[j, k]
        end_time = time.time()
        self.matrix_time = end_time-start_time

def eval_allo_test():
    #test
    n_x = 156
    n_y = 156
    resolution_x = 0.05
    resolution_y = 0.05
    mapset_dir = '/home/data/wzr/no_com_1/data/allocation_exp_test_2/batch_5'
    norm_factor = 1/ 156 / 0.05
    eval_data = evalData(mapset_dir,norm_factor)
    # eval_data.calcu_time(n_x,n_y,resolution_x,resolution_y)
    # print(eval_data.matrix_time)

    #debug
    # print(eval_data.n_robot,eval_data.n_task,eval_data.n_obstacle)
    # print(eval_data.costmats[0])

    #net
    net_r_f = torch.tensor(eval_data.feature_robot[0],dtype=torch.float32).unsqueeze(0)
    net_t_f = torch.tensor(eval_data.feature_task[0],dtype=torch.float32).unsqueeze(0)
    net_o_f = torch.tensor(eval_data.feature_obstacle[0],dtype=torch.float32).unsqueeze(0)
    net_c_m = torch.tensor(eval_data.costmats[0],dtype=torch.float32).unsqueeze(0)
    eval_net = allocation.AllocationNet(4, 128, 1, 8 )
    model_path = '/home/data/wzr/no_com_1/model/allocation/2024_12_12_exp_0/time_2024-12-13-11-11-11_dis_23.44.pt'
    eval_net.load_state_dict(torch.load(model_path))
    eval_net.eval()
    time_start = time.time()
    with torch.no_grad():
        eval_net.config(eval_data.cfg)
        seq, pro, distance_net = eval_net(net_r_f,net_t_f,net_o_f,net_c_m)
    time_end = time.time()
    print('time cost',time_end-time_start,'s')
    print(distance_net)
    print(seq)

    #greedy
    greedy_rt_mat = eval_data.costmats[0, :eval_data.n_robot, eval_data.n_robot:] #(n_robot,n_task)
    greedy_tt_mat = eval_data.costmats[0, eval_data.n_robot:, eval_data.n_robot:] #(n_task,n_task)
    eval_greedy = greedy.GreedyTaskAllocationPlanner()
    time_start = time.time()
    schedules_greedy = eval_greedy.greedy_allocate_mat(greedy_rt_mat, greedy_tt_mat)
    time_end = time.time()
    print('time cost',time_end-time_start,'s')
    distance_greedy = eval_greedy.allocation_distance_eval_mat(greedy_rt_mat, greedy_tt_mat, schedules_greedy)
    print(distance_greedy)
    print(schedules_greedy)

    # schedules_net = [[4],[6,0,2],[1],[],[3,5]]
    # distance_net = eval_greedy.allocation_distance_eval_mat(greedy_rt_mat, greedy_tt_mat, schedules_net)
    # print(distance_net)

    #ga
    eval_ga = ga.GATaskAllocationPlanner()
    # planner.config_time_estimate_function(planner.pose_distance)
    time_start = time.time()
    schedules_ga = eval_ga.ga_allocate_mat(greedy_rt_mat, greedy_tt_mat)
    time_end = time.time()
    print('time cost',time_end-time_start,'s')
    distance_ga = eval_ga.allocation_distance_eval_mat(greedy_rt_mat, greedy_tt_mat, schedules_ga)
    print(distance_ga)


def read_time_cost():
    mapset_dir = '/home/data/wzr/no_com_1/data/allocation_compare_1'
    csv_dir = '/home/data/wzr/no_com_1/log/exp_result'
    n_batch = 64
    batch_size = 100
    save_time = np.zeros((n_batch,batch_size),dtype=float)
    for i in range(n_batch):
        batch_dir = os.path.join(mapset_dir, f"batch_{i}")
        for j in range(batch_size):
            map_path = os.path.join(batch_dir, f"map_{j}")
            json_path = os.path.join(map_path, "map_info.json")
            with open(json_path, "r") as f:
                map_info = json.load(f)
            save_time[i,j] = map_info["time_cost"]
    with open(os.path.join(csv_dir, "time_cost.csv"), "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["batch_id","map_id","time"])
        for i in range(n_batch):
            for j in range(batch_size):
                writer.writerow([i,j,save_time[i,j]])


def eval_net():
    DEVICE = torch.device("cpu")
    n_x = 156
    n_y = 156
    resolution_x = 0.05
    resolution_y = 0.05
    norm_factor = 1/ 156 / 0.05
    mapset_dir = '/home/data/wzr/no_com_1/data/allocation_compare_1'
    csv_dir = '/home/data/wzr/no_com_1/log/exp_result'
    #读取batch_size信息json文件
    dataset_info = os.path.join(mapset_dir, "dataset_info.json")
    with open(dataset_info, "r") as f:
        dataset_info = json.load(f)

    # batch_size = dataset_info["batch_size"]
    batch_size = 100
    n_batch = dataset_info["n_batch"]
    ob_points = dataset_info["ob_points"]
    
    batchs = []
    for i in range(n_batch):
        dir = os.path.join(mapset_dir, f"batch_{i}")
        batchs.append(evalData(dir,norm_factor))

    eval_net = allocation.AllocationNet(4, 128, 1, 8 )
    model_path = '/home/data/wzr/no_com_1/model/allocation/2024_12_12_exp_0/time_2024-12-13-11-11-11_dis_23.44.pt'
    eval_net.load_state_dict(torch.load(model_path))
    eval_net.eval()

    save_n_r = np.zeros((n_batch),dtype=int)
    save_n_t = np.zeros((n_batch),dtype=int)
    save_n_o = np.zeros((n_batch),dtype=int)
    save_time = np.zeros((n_batch,batch_size),dtype=float)
    save_distance = np.zeros((n_batch,batch_size),dtype=float)

    for i in range(n_batch):
        print(f"batch {i}")
        for j in range(batch_size):
            net_r_f = torch.tensor(batchs[i].feature_robot[j],dtype=torch.float32).unsqueeze(0)
            net_t_f = torch.tensor(batchs[i].feature_task[j],dtype=torch.float32).unsqueeze(0)
            net_o_f = torch.tensor(batchs[i].feature_obstacle[j],dtype=torch.float32).unsqueeze(0)
            net_c_m = torch.tensor(batchs[i].costmats[j],dtype=torch.float32).unsqueeze(0)
            eval_net.config(batchs[i].cfg)
            time_start = time.time()
            with torch.no_grad():
                seq, pro, distance_net = eval_net(net_r_f,net_t_f,net_o_f,net_c_m)
            time_end = time.time()
            save_time[i,j] = time_end-time_start
            save_distance[i,j] = distance_net

        save_n_r[i] = batchs[i].n_robot
        save_n_t[i] = batchs[i].n_task
        save_n_o[i] = batchs[i].n_obstacle
    
    with open(os.path.join(csv_dir, "eval_net.csv"), "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["batch_id","map_id","n_robot","n_task","n_obstacle","time","distance"])
        for i in range(n_batch):
            for j in range(batch_size):
                writer.writerow([i,j,save_n_r[i],save_n_t[i],save_n_o[i],save_time[i,j],save_distance[i,j]])
    print("eval_net done")

def eval_greedy():
    n_x = 156
    n_y = 156
    resolution_x = 0.05
    resolution_y = 0.05
    norm_factor = 1/ 156 / 0.05
    mapset_dir = '/home/data/wzr/no_com_1/data/allocation_compare_1'
    csv_dir = '/home/data/wzr/no_com_1/log/exp_result'
    #读取batch_size信息json文件
    dataset_info = os.path.join(mapset_dir, "dataset_info.json")
    with open(dataset_info, "r") as f:
        dataset_info = json.load(f)

    # batch_size = dataset_info["batch_size"]
    batch_size = 100
    n_batch = dataset_info["n_batch"]
    ob_points = dataset_info["ob_points"]
    
    batchs = []
    for i in range(n_batch):
        dir = os.path.join(mapset_dir, f"batch_{i}")
        batchs.append(evalData(dir,norm_factor))

    eval_greedy = greedy.GreedyTaskAllocationPlanner()

    save_n_r = np.zeros((n_batch),dtype=int)
    save_n_t = np.zeros((n_batch),dtype=int)
    save_n_o = np.zeros((n_batch),dtype=int)
    save_time = np.zeros((n_batch,batch_size),dtype=float)
    save_distance = np.zeros((n_batch,batch_size),dtype=float)

    for i in range(n_batch):
        print(f"batch {i}")
        for j in range(batch_size):
            greedy_rt_mat = batchs[i].costmats[j, :batchs[i].n_robot, batchs[i].n_robot:] #(n_robot,n_task)
            greedy_tt_mat = batchs[i].costmats[j, batchs[i].n_robot:, batchs[i].n_robot:] #(n_task,n_task)
            time_start = time.time()
            schedules_greedy = eval_greedy.greedy_allocate_mat(greedy_rt_mat, greedy_tt_mat)
            time_end = time.time()
            distance_greedy = eval_greedy.allocation_distance_eval_mat(greedy_rt_mat, greedy_tt_mat, schedules_greedy)
            save_time[i,j] = time_end-time_start
            save_distance[i,j] = distance_greedy

        save_n_r[i] = batchs[i].n_robot
        save_n_t[i] = batchs[i].n_task
        save_n_o[i] = batchs[i].n_obstacle

    with open(os.path.join(csv_dir, "eval_greedy.csv"), "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["batch_id","map_id","n_robot","n_task","n_obstacle","time","distance"])
        for i in range(n_batch):
            for j in range(batch_size):
                writer.writerow([i,j,save_n_r[i],save_n_t[i],save_n_o[i],save_time[i,j],save_distance[i,j]])
    print("eval_greedy done")

def eval_ga():
    n_x = 156
    n_y = 156
    resolution_x = 0.05
    resolution_y = 0.05
    norm_factor = 1/ 156 / 0.05
    mapset_dir = '/home/data/wzr/no_com_1/data/allocation_compare_1'
    csv_dir = '/home/data/wzr/no_com_1/log/exp_result'
    #读取batch_size信息json文件
    dataset_info = os.path.join(mapset_dir, "dataset_info.json")
    with open(dataset_info, "r") as f:
        dataset_info = json.load(f)

    # batch_size = dataset_info["batch_size"]
    batch_size = 100
    n_batch = dataset_info["n_batch"]
    ob_points = dataset_info["ob_points"]
    
    batchs = []
    for i in range(n_batch):
        dir = os.path.join(mapset_dir, f"batch_{i}")
        batchs.append(evalData(dir,norm_factor))

    save_n_r = np.zeros((n_batch),dtype=int)
    save_n_t = np.zeros((n_batch),dtype=int)
    save_n_o = np.zeros((n_batch),dtype=int)
    save_time = np.zeros((n_batch,batch_size),dtype=float)
    save_distance = np.zeros((n_batch,batch_size),dtype=float)

    for i in range(n_batch):
        save_n_r[i] = batchs[i].n_robot
        save_n_t[i] = batchs[i].n_task
        save_n_o[i] = batchs[i].n_obstacle

    n_workers = 64
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i in range(n_batch):
            for j in range(batch_size):
                ga_rt_mat = batchs[i].costmats[j, :batchs[i].n_robot, batchs[i].n_robot:]
                ga_tt_mat = batchs[i].costmats[j, batchs[i].n_robot:, batchs[i].n_robot:]
                futures.append(executor.submit(_ga_process, i, j, ga_rt_mat, ga_tt_mat))
        counter = 0
        for future in futures:
            counter += 1
            batch_idx, map_idx, used_time, distance_ga = future.result()
            save_time[batch_idx,map_idx] = used_time
            save_distance[batch_idx,map_idx] = distance_ga
            if counter % 100 == 0:
                print(f"counter: {counter}")

    with open(os.path.join(csv_dir, "eval_ga.csv"), "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["batch_id","map_id","n_robot","n_task","n_obstacle","time","distance"])
        for i in range(n_batch):
            for j in range(batch_size):
                writer.writerow([i,j,save_n_r[i],save_n_t[i],save_n_o[i],save_time[i,j],save_distance[i,j]])
    print("eval_ga done")

def _ga_process(batch_idx, map_idx,rt_mat,tt_mat):
    eval_ga = ga.GATaskAllocationPlanner()
    time_start = time.time()
    schedules_ga = eval_ga.ga_allocate_mat(rt_mat, tt_mat)
    time_end = time.time()
    used_time = time_end - time_start
    distance_ga = eval_ga.allocation_distance_eval_mat(rt_mat, tt_mat, schedules_ga)
    return batch_idx, map_idx, used_time, distance_ga

if __name__ == "__main__":
    # eval_greedy()
    # eval_ga()
    # eval_net()
    read_time_cost()
    

