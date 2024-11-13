import sys
sys.path.append('./net')
import math
import os
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
import allocation
import intention
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
        # print('arrived')
        return True
    else:
        return False
    
def test_robot_2():
    #config
    n_robot = 2
    n_task = 5
    n_obstacle = 3
    n_x = 100
    n_y = 100
    resolution_x = 0.1
    resolution_y = 0.1

    arrival_r = 0.1
    sim_steps = 1000
    dt = 0.1

    seed = 2
    batch_size = 1

    config_dir = './config'

    #map
    map_data = mp.Map(n_obstacle, n_robot, n_task, n_x, n_y, resolution_x, resolution_y)
    rng = np.random.default_rng(seed)
    map_data.setObstacleRandn(rng)
    # map_data.plot()
    # map_data.plotGrid()

    #state
    robot_state = []
    task_state = []
    for i in range(n_robot):
        state = np.array([map_data.starts[i][0]*map_data.n_x*map_data.resolution_x, 
                          map_data.starts[i][1]*map_data.n_y*map_data.resolution_y, 
                          -math.pi, 0.0, 0.0])
        robot_state.append(state)
    for i in range(n_task):
        state = np.array([map_data.tasks[i][0]*map_data.n_x*map_data.resolution_x, 
                          map_data.tasks[i][1]*map_data.n_y*map_data.resolution_y, 
                          0])
        task_state.append(state)

    #robot
    cfg_0 = os.path.join(config_dir, 'robot_0.json')
    robot_0 = Robot(map_data, cfg_0)
    cfg_1 = os.path.join(config_dir, 'robot_1.json')
    robot_1 = Robot(map_data, cfg_1)    

    #sim
    for i_step in range(sim_steps):
        #arrived
        task_finish_count = 0
        for j in range(n_task):
            if task_state[j][2] == 1:
                task_finish_count += 1
                continue
            for k in range(n_robot):
                if _judge_arrival(robot_state[k], task_state[j], arrival_r):
                    task_state[j][2] = 1
                    print('robot', k, 'arrived task', j)
                    task_finish_count += 1
                    break
        if task_finish_count == n_task:
            print('all tasks finished')
            break
    

        #robot
        out_0 = robot_0.base_callback(robot_state,task_state)
        out_1 = robot_1.base_callback(robot_state,task_state)

        #update robot state
        robot_state[0] = _state_update(robot_state[0], out_0, dt)
        robot_state[1] = _state_update(robot_state[1], out_1, dt)

        print('robot_0:', robot_state[0], 'robot_1:', robot_state[1])
            
  

if __name__ == "__main__":
    test_robot_2()