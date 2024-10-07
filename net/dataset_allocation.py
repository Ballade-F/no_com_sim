import os
import numpy as np
import json
import csv
import torch
from torch.utils.data import Dataset, DataLoader




class batchData():
    def __init__(self, dir):
        batch_info = os.path.join(dir, "batch_info.json")
        with open(batch_info, "r") as f:
            self.batch_info = json.load(f)
        self.n_robot = self.batch_info["n_robot"]
        self.n_task = self.batch_info["n_task"]
        self.n_obstacle = self.batch_info["n_obstacle"]
        self.batch_size = self.batch_info["batch_size"]
        self.ob_points = self.batch_info["ob_points"]

        self.feature_robot = np.full((self.batch_size, self.n_robot, 3),-1, dtype=float)
        self.feature_task = np.full((self.batch_size, self.n_task, 3),-1, dtype=float) 
        self.feature_obstacle = np.full((self.batch_size, self.n_obstacle, self.ob_points, 2),-1, dtype=float)
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
                        idx_point = (idx-self.n_robot-self.n_task-1)-idx_ob*self.ob_points
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
        self.ob_points = self.dataset_info["ob_points"]
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
        #到robot的cost设置成0
        costmat[:,:,:self.batchs[idx].n_robot] = 0
        cfg = {
            "n_robot": int(self.batchs[idx].n_robot),
            "n_task": int(self.batchs[idx].n_task),
            "n_obstacle": int(self.batchs[idx].n_obstacle),
            "ob_points": int(self.batchs[idx].ob_points),
            "batch_size": int(self.batchs[idx].batch_size)
        }
        return feature_robot, feature_task, feature_obstacle, costmat, cfg


if __name__ == '__main__':
    dataset = AllocationDataset("allocation_data", 10)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (feature_robot, feature_task, feature_obstacle, costmat,cfg) in enumerate(dataloader):
        print(feature_robot.shape)
        print(feature_task.shape)
        print(feature_obstacle.shape)
        print(costmat.shape)
        print(cfg)
        break

    

                
