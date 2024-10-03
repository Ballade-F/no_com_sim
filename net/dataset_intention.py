import os
import json
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class IntentionDataset(Dataset):
    def __init__(self, map_dirs, n_map, n_robot, n_task):
        self.map_dirs = map_dirs
        self.n_map = n_map
        self.n_robot = n_robot
        self.n_task = n_task
        self.map_items_num = []
        self.len = 0
        self.traj_items_num = 5
        for i in range(n_map):
            map_dir = os.path.join(self.map_dirs, f"map_{i}")
            with open(os.path.join(map_dir, "map_info.json"), "r") as f:
                map_info = json.load(f)
                len = (map_info["length"]+1-self.traj_items_num) if map_info["length"] >= self.traj_items_num else 0
                self.map_items_num.append(len)
                self.len += len

    
    def __len__(self):
        return sum(self.map_items_num)
    
    def __getitem__(self, idx):
        map_idx = 0
        if idx >= self.len:
            raise IndexError("Index out of range")
        while idx >= self.map_items_num[map_idx]:
            idx -= self.map_items_num[map_idx]
            map_idx += 1
        
        map_dir = os.path.join(self.map_dirs, f"map_{map_idx}")
        trajectory_path = os.path.join(map_dir, "trajectory.csv")
        trajectory_df = pd.read_csv(trajectory_path)

        # x,y in 5 time steps
        traj_robot = np.zeros((self.n_robot,2*self.traj_items_num))
        # label
        label = np.zeros((self.n_robot),dtype=int)
        # x,y,finish_flag
        # 最后一个task是虚拟task，用于结束任务的robot
        feature_task = np.zeros((self.n_task+1,3))
        feature_task[-1,0] = -1
        feature_task[-1,1] = -1
        feature_task[-1,2] = 0
        
        # Extract data starting from idx for traj_items_num time points
        end_idx = idx + self.traj_items_num
        for i in range(idx, end_idx):
            for item in trajectory_df.loc[trajectory_df['Time'] == i].values:
                if item[1] == 'robot':
                    traj_robot[int(item[2]), 2*(i-idx):2*(i-idx)+2] = item[3:5]
                    # Last time point is used for label
                    if i == end_idx-1:
                        label[int(item[2])] = item[5] if item[5] != -1 else self.n_task
                elif item[1] == 'task' and i == end_idx-1:
                    feature_task[int(item[2]),:] = item[3:6]
                    # feature_task[int(item[2]),2] = 1.0 if item[5] == True else 0.0
                    

        feature_robot = torch.FloatTensor(traj_robot)
        label = torch.LongTensor(label)
        feature_task = torch.FloatTensor(feature_task)

        return feature_robot, label, feature_task


        

# Test function
def test_IntentionDataset():
    map_dirs = "intention_data"
    n_map = 3
    n_robot = 5
    n_task = 5
    dataset = IntentionDataset(map_dirs, n_map, n_robot, n_task)
    print("Dataset Length:", len(dataset))
    print("last data", dataset[74])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    for batch in dataloader:
        feature_robot, label, feature_task = batch
        print("Feature Robot Shape:", feature_robot.shape)
        print("Label Shape:", label.shape)
        print("Feature Task Shape:", feature_task.shape)
        assert feature_robot.shape == (4, n_robot, 2 * dataset.traj_items_num)
        assert label.shape == (4, n_robot)
        assert feature_task.shape == (4, n_task+1, 3)
        break  # Test only the first batch

if __name__ == "__main__":
    test_IntentionDataset()