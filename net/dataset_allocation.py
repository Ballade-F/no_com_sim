import os
import numpy as np
import json
import csv
import torch
from torch.utils.data import Dataset, DataLoader
import utils.map as mp
from pnc.path_planner import AStarPlanner



class batchData():
    def __init__(self, map, n_robot, n_task, n_obstacle):
        self.map = map
        self.n_robot = n_robot
        self.n_task = n_task
        self.n_obstacle = n_obstacle
        self.costmats = np.zeros((n_robot+n_task, n_robot+n_task))
        self.maps = []
        for i in range(n_robot+n_task):
            self.maps.append(map.deepcopy())
        self.costmats = self.calCostmat()
            
        
        

#出于批量训练的考虑，一个batch中的规模是固定的
class AllocationDataset(Dataset):
    def __init__(self, map_dirs, n_batch):
        super(AllocationDataset, self).__init__()
        #读取batch信息json文件
        batch_info = os.path.join(map_dirs, "batch_info.json")
        with open(batch_info, "r") as f:
            self.batch_info = json.load(f)
        self.batch_size = self.batch_info["batch_size"]

        #读取各个batch


        
            


    def __len__(self):
        return self.n_batch
    
    def __getitem__(self, idx):
        if idx >= self.n_batch:
            raise IndexError("Index out of range")
        robot_feature = np.zeros((self.batch_size, self.batchs[idx].n_robot, 3)) # x,y,node_flag 0:robot 1:task -1:obstacle
        task_feature = np.zeros((self.batch_size, self.batchs[idx].n_task, 3))
        obstacle_feature = np.zeros((self.batch_size, self.batchs[idx].n_obstacle, mp.n_ob_points, 2))
        costmat = self.batchs[idx].costmats
        for i in range(self.batch_size):
            map = self.batchs[idx].maps[i]
            robot_feature[i,:,:2] = map.starts_grid
            task_feature[i,:,:2] = map.tasks_grid
            robot_feature[i,:,2] = 0
            task_feature[i,:,2] = 1
            for j in range(map.n_obstacles):
                obstacle_feature[i,j,:,:] = map.obstacles[j]
        return robot_feature, task_feature, obstacle_feature, costmat
    


if __name__ == '__main__':
    dataset = AllocationDataset(2, 2, 0)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, data in enumerate(dataloader):
        robot_feature, task_feature, obstacle_feature, costmat = data
        print(robot_feature)
        print(task_feature)
        print(obstacle_feature)
        print(costmat)
        print("batch", i)
        print("robot_feature", robot_feature.shape)
        print("task_feature", task_feature.shape)
        print("obstacle_feature", obstacle_feature.shape)
        print("costmat", costmat.shape)

    

                
