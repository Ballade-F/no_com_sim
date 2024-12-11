import os
import json
import csv
import numpy as np
from math import sqrt
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd




#与allocation不同，一个实例只提取一个scale的数据，batchsize就可以随意设置了
class IntentionDataset(Dataset):
    def __init__(self, scale_dirs, n_robot_points):
        super(IntentionDataset, self).__init__()
        self.n_robot_points = n_robot_points
        scale_info_dir = os.path.join(scale_dirs, "scale_info.json")
        # 读取scale信息json文件
        with open(scale_info_dir, "r") as f:
            self.scale_info = json.load(f)
        self.n_robot = self.scale_info["n_robot"]
        self.n_task = self.scale_info["n_task"]
        self.n_obstacle = self.scale_info["n_obstacle"]
        self.n_map = self.scale_info["n_map"]
        self.ob_points = self.scale_info["ob_points"]
        self.resolution_x = self.scale_info["resolution_x"]
        self.resolution_y = self.scale_info["resolution_y"]
        self.n_x = self.scale_info["n_x"]
        self.n_y = self.scale_info["n_y"]
        self.cfg = {'n_robot': self.n_robot, 
                    'n_task': self.n_task, 
                    'n_obstacle': self.n_obstacle, 
                    'ob_points': self.ob_points}

        
        self.traj_robot = []
        self.traj_task = []
        self.traj_len = []
        self.feature_obstacle = np.full((self.n_map, self.n_obstacle, self.ob_points, 2),-1, dtype=float)
        for i_map in range(self.n_map):
            map_dir = os.path.join(scale_dirs, f"map_{i_map}")
            trajectory_path = os.path.join(map_dir, "trajectory.csv")
            trajectory_df = pd.read_csv(trajectory_path)
            map_traj_robot = []
            map_traj_task = []
            map_len = trajectory_df["Time"].max()+1 - (n_robot_points-1)  #实际每个map能用的样本数
            self.traj_len.append(map_len)
            for i in range(trajectory_df["Time"].max()+1):
                traj_robot = np.zeros((self.n_robot,3))
                traj_task = np.zeros((self.n_task,3)) 
                for item in trajectory_df.loc[trajectory_df['Time'] == i].values:
                    if item[1] == 'robot':
                        traj_robot[int(item[2]),:] = item[3:]
                    elif item[1] == 'task':
                        traj_task[int(item[2]),:] = item[3:]
                map_traj_robot.append(traj_robot)
                map_traj_task.append(traj_task)
            self.traj_robot.append(map_traj_robot)
            self.traj_task.append(map_traj_task)

            # 读取障碍物信息
            info_path = os.path.join(map_dir, "info.csv")
            with open(info_path, "r") as f:
                reader = csv.reader(f)
                for idx, row in enumerate(reader):
                    #表头
                    if idx <= self.n_robot+self.n_task:
                        continue
                    idx_ob = int(row[0])-1
                    idx_point = (idx-self.n_robot-self.n_task-1)-idx_ob*self.ob_points
                    self.feature_obstacle[i_map, idx_ob, idx_point,:] = row[1:]

    def __len__(self):
        return sum(self.traj_len)
    
    #return: feature_robot, label, feature_task, feature_obstacle
    #feature_robot:(n_robot,r_points+2, 2), r_points为[-1,1]归一化坐标，后面为平均位置，平均速率
    #feature_obstacle:(n_obstacle, ob_points, 2)
    def __getitem__(self, idx):
        map_idx = 0
        if idx >= self.__len__():
            raise IndexError("Index out of range")
        while idx >= self.traj_len[map_idx]:
            idx -= self.traj_len[map_idx]
            map_idx += 1
        
        traj_robot_item = np.zeros((self.n_robot,self.n_robot_points,2))
        label = np.zeros((self.n_robot),dtype=int)
        traj_task = np.zeros((self.n_task+1,3))#最后一个task是虚拟task，用于结束任务的robot
        traj_task[-1,0] = -1
        traj_task[-1,1] = -1
        traj_task[-1,2] = 0
        for i in range(self.n_robot_points):
            traj_robot_item[:,i,:] = self.traj_robot[map_idx][idx+i][:,:2]
            if i == self.n_robot_points-1:
                for j in range(self.n_robot):
                    label[j] = self.traj_robot[map_idx][idx+i][j,2] if self.traj_robot[map_idx][idx+i][j,2] != -1 else self.n_task
        for i in range(self.n_task):
            traj_task[i,:] = self.traj_task[map_idx][idx+self.n_robot_points-1][i]

        robot_ave = np.mean(traj_robot_item, axis=1)#(n_robot,2) 平均位置
        robot_delta = np.zeros((self.n_robot,self.n_robot_points-1,2))
        for i in range(self.n_robot_points-1):
            robot_delta[:,i,:] = abs(traj_robot_item[:,i+1,:] - traj_robot_item[:,i,:])
        robot_delta = np.mean(robot_delta, axis=1) #(n_robot,2) 平均速率,按最大速率归一化
        robot_delta[:,0] = robot_delta[:,0]*self.n_x
        robot_delta[:,1] = robot_delta[:,1]*self.n_y
        # 归一化坐标到 [-1, 1]
        traj_robot_norm = np.zeros((self.n_robot, self.n_robot_points, 2))
        for i in range(self.n_robot):
            min_val = np.min(traj_robot_item[i], axis=0) # (2,)
            max_val = np.max(traj_robot_item[i], axis=0) # (2,)
            traj_robot_norm[i] = 2 * (traj_robot_item[i] - min_val) / (max_val - min_val + 1e-6) - 1

        # 拼接 traj_robot_norm, robot_ave, robot_delta
        traj_robot_return = np.concatenate((traj_robot_norm, robot_ave[:, np.newaxis, :], robot_delta[:, np.newaxis, :]), axis=1)
        
        # 处理 obstacle
        traj_obstacle_item = self.feature_obstacle[map_idx] # (n_obstacle, ob_points, 2)
        # obstacle_ave = np.mean(traj_obstacle_item, axis=1) # (n_obstacle, 2)
        # obstacle_min = np.min(traj_obstacle_item, axis=1) # (n_obstacle, 2)
        # obstacle_max = np.max(traj_obstacle_item, axis=1) # (n_obstacle, 2)
        # obstacle_delta = obstacle_max - obstacle_min + 1e-6  # 防止除零
        # # 归一化坐标到 [-1, 1]
        # # traj_obstacle_norm = np.zeros((self.n_obstacle, self.ob_points, 2))
        # traj_obstacle_norm = 2 * (traj_obstacle_item - obstacle_min[:, np.newaxis, :]) / obstacle_delta[:, np.newaxis, :] - 1 # (n_obstacle, ob_points, 2)
        # # 拼接 traj_obstacle_norm, obstacle_ave, obstacle_delta
        # traj_obstacle_return = np.concatenate((traj_obstacle_norm, obstacle_ave[:, np.newaxis, :], obstacle_delta[:, np.newaxis, :]), axis=1) # (n_obstacle, ob_points+2, 2)
        
        # 转换为 PyTorch 张量
        feature_robot = torch.FloatTensor(traj_robot_return)
        label = torch.LongTensor(label)
        feature_task = torch.FloatTensor(traj_task)
        feature_obstacle = torch.FloatTensor(traj_obstacle_item)

        return feature_robot, label, feature_task, feature_obstacle
        # feature_robot: (n_robot, n_robot_points+2, 2) 
        # label: (n_robot)
        # feature_task: (n_task+1, 3)
        # feature_obstacle: (n_obstacle, ob_points, 2)

        

# Test function
def test_IntentionDataset():
    map_dirs = "/home/ballade/Desktop/Project/no_com_sim/intention_data/scale_0"
    n_point_num = 5
    dataset = IntentionDataset(map_dirs, n_point_num)
    print("Dataset Length:", len(dataset))
    print("last data", dataset[36])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    for batch in dataloader:
        feature_robot, label, feature_task, feature_obstacle = batch
        print("Feature Robot Shape:", feature_robot.shape)
        print("Label Shape:", label.shape)
        print("Feature Task Shape:", feature_task.shape)
        print("Feature Obstacle Shape:", feature_obstacle.shape)
        break  # Test only the first batch

if __name__ == "__main__":
    test_IntentionDataset()