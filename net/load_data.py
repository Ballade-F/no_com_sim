import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class MapDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.map_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    def __len__(self):
        return len(self.map_dirs)
    
    def __getitem__(self, idx):
        map_dir = self.map_dirs[idx]
        
        # Load map size
        with open(os.path.join(map_dir, "map_size.txt"), "r") as f:
            map_size = int(f.readline().strip().split(": ")[1])
        
        # Load trajectory data
        trajectory = pd.read_csv(os.path.join(map_dir, "trajectory.csv")).values
        
        sample = {
            'map_size': map_size,
            'trajectory': torch.tensor(trajectory, dtype=torch.float32)
        }
        
        return sample

# Usage
data_dir = 'intention_data'
dataset = MapDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Example of iterating through the DataLoader
for batch in dataloader:
    print(batch)