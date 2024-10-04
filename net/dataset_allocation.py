import os
import numpy as np
import json
import csv
import torch
from torch.utils.data import Dataset, DataLoader
import utils.map as mp
from pnc.path_planner import AStarPlanner



class batchData():
    def __init__(self, dir):
        batch_info = os.path.join(dir, "batch_info.json")
        with open(batch_info, "r") as f:
            self.batch_info = json.load(f)
        self.n_robot = self.batch_info["n_robot"]
        self.n_task = self.batch_info["n_task"]
        self.n_obstacle = self.batch_info["n_obstacle"]
        self.batch_size = self.batch_info["batch_size"]

        self.feature_robot = np.full((self.batch_size, self.n_robot, 3),-1, dtype=float)
        self.feature_task = np.full((self.batch_size, self.n_task, 3),-1, dtype=float) 
        self.feature_obstacle = np.full((self.batch_size, self.n_obstacle, mp.n_ob_points, 2),-1, dtype=float)
        self.costmats = np.full((self.batch_size, self.n_robot+self.n_task, self.n_robot+self.n_task),-1, dtype=float)

        for i in range(self.batch_size):
            map_dir = os.path.join(dir, f"map_{i}")
            costmat = np.load(os.path.join(map_dir, "costmat.npy"))
            self.costmats[i,:,:] = costmat
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
                        idx_point = (idx-self.n_robot-self.n_task-1)-idx_ob*mp.n_ob_points
                        self.feature_obstacle[i, idx_ob, idx_point,:] = row[1:]

            

        
            
        
        

#出于批量训练的考虑，一个batch中的规模是固定的
class AllocationDataset(Dataset):
    def __init__(self, dataset_dir, n_batch):
        super(AllocationDataset, self).__init__()

        #读取batch_size信息json文件
        dataset_info = os.path.join(dataset_dir, "dataset_info.json")
        with open(dataset_info, "r") as f:
            self.dataset_info = json.load(f)

        self.batch_size = self.dataset_info["batch_size"]
        n_batch_max = self.dataset_info["n_batch"]
        if n_batch > n_batch_max:
            raise ValueError("n_batch exceeds the maximum value")

        #读取各个batch
        self.batchs = []
        self.n_batch = n_batch
        for i in range(n_batch):
            dir = os.path.join(dataset_dir, f"batch_{i}")
            self.batchs.append(batchData(dir))
            

    def __len__(self):
        return self.n_batch
    
    def __getitem__(self, idx):
        if idx >= self.n_batch:
            raise IndexError("Index out of range")
        feature_robot = torch.from_numpy(self.batchs[idx].feature_robot).float()
        feature_task = torch.from_numpy(self.batchs[idx].feature_task).float()
        feature_obstacle = torch.from_numpy(self.batchs[idx].feature_obstacle).float()
        costmat = torch.from_numpy(self.batchs[idx].costmats).float()
        return feature_robot, feature_task, feature_obstacle, costmat


if __name__ == '__main__':
    pass

    

                
