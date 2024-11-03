import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import utils.map as mp
import utils.ringbuffer as ringbuffer
import pnc.dwa as dwa
import pnc.path_planner as path_planner
import task_allocation.hungarian as hungarian
import task_allocation.greedy_allocation_lib as greedy
import task_allocation.ga_allocation_lib as ga
import net.allocation as allocation
import net.intention as intention
from robot import Robot



def _state_update(state,u,dt):
    state[0] += u[0]*np.cos(state[2])*dt
    state[1] += u[0]*np.sin(state[2])*dt
    state[2] += u[1]*dt
    state[3] = u[0]
    state[4] = u[1]
    return state

def _judge_arrival(state, target, r):
    if np.linalg.norm(state[:2]-target[:2]) < r:
        print('arrived')
        return True
    else:
        return False
    
def test_robot_2():
    pass