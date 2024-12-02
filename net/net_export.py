import torch
import torchvision
import os
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from intention import IntentionNet
from dataset_intention import IntentionDataset
from allocation import AllocationNet
from dataset_allocation import AllocationDataset
import json
import time as TM
import logging

def intention_export():
    intention_model_dir = '/home/data/wzr/no_com_1/model/intention/time_2024-12-01-12-06-51_acc_93.22.pt'
    intention_model = IntentionNet(16, 5, 128, 128, 8)
    intention_model.load_state_dict(torch.load(intention_model_dir))
    # intention_model.batch_size = 1
    # intention_model.robot_n = 2
    # intention_model.task_n = 3
    # intention_model.ob_n = 1
    # intention_model.ob_points = 4
    intention_model.r_points = 5
    
    x_r = torch.randn(1, 2, 5, 2)
    x_t = torch.randn(1, 5, 3)
    x_ob = torch.randn(1, 1, 4, 2)
    is_train = torch.tensor(False)
    
    # example = x_r, x_t, x_ob, is_train
    traced_script_module = torch.jit.script(intention_model)
    traced_script_module.config_script(2, 4, 1, 1, 4)
    output = traced_script_module(x_r, x_t, x_ob, is_train)
    print(output.shape)
    # traced_script_module.save("intention_model.pt")
    
def allocation_export():
    # allocation_model_dir
    allocation_model = AllocationNet(16, 128, 128, 8)
    # allocation_model.load_state_dict(torch.load(allocation_model_dir))
    allocation_model.config_export()
    traced_script_module = torch.jit.script(allocation_model)
    traced_script_module.config_script(2, 5, 1, 1, 4)
    x_r = torch.randn(1, 2, 3)
    x_t = torch.randn(1, 5, 3)
    x_ob = torch.randn(1, 1, 4, 2)
    nothing = torch.randn(1, 2, 3)
    output = traced_script_module(x_r, x_t, x_ob,nothing)
    print(output)
    # traced_script_module.save("allocation_model.pt")
    

    # allocation_model = AllocationNet(4, 128, 1, 8)
    # allocation_model.config_export()
    # allocation_model.config_script(2, 5, 1, 1, 4)
    # x_r = torch.randn(1, 2, 3)
    # x_t = torch.randn(1, 5, 3)
    # x_ob = torch.randn(1, 1, 4, 2)
    # output = allocation_model(x_r, x_t, x_ob,None)
    # print(output)

if __name__ == "__main__":
    # intention_export()
    allocation_export()
    
    
    