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
import matplotlib.pyplot as plt




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
    sim_steps = 100
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

    #state 真实坐标
    robot_state = np.zeros((n_robot,5))
    task_state = np.zeros((n_task,3))
    for i in range(n_robot):
        robot_state[i] = np.array([map_data.starts[i][0]*map_data.n_x*map_data.resolution_x, 
                          map_data.starts[i][1]*map_data.n_y*map_data.resolution_y, 
                          -math.pi, 0.0, 0.0])
    for i in range(n_task):
        task_state[i] = np.array([map_data.tasks[i][0]*map_data.n_x*map_data.resolution_x, 
                          map_data.tasks[i][1]*map_data.n_y*map_data.resolution_y, 
                          0])
    # print('robot_state:', robot_state)
    # print('task_state:', task_state)

    #robot
    cfg_0 = os.path.join(config_dir, 'robot_0.json')
    robot_0 = Robot(map_data, cfg_0,robot_state[:,:3],task_state)
    cfg_1 = os.path.join(config_dir, 'robot_1.json')
    robot_1 = Robot(map_data, cfg_1,robot_state[:,:3],task_state)    

    #sim
    path_robot_0 = []
    for i_step in range(sim_steps):
        #arrived 真实坐标
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
        out_0 = robot_0.base_callback(robot_state[:,:3],task_state)
        # out_1 = robot_1.base_callback(robot_state,task_state)

        #update robot state
        robot_state[0] = _state_update(robot_state[0], out_0, dt)
        # robot_state[1] = _state_update(robot_state[1], out_1, dt)

        path_robot_0.append(robot_state[0,:2].copy())
        print('robot_0:', robot_state[0], 'target:', robot_0.task_list)

    #plot
    fig, ax = plt.subplots()
    ax.set_xlim(0, map_data.n_x*map_data.resolution_x)
    ax.set_ylim(0, map_data.n_y*map_data.resolution_y)
    for ob_points in map_data.obstacles:
        ax.fill(map_data.n_x*map_data.resolution_x*ob_points[:, 0], map_data.n_y*map_data.resolution_y*ob_points[:, 1], 'g')
    for i in range(map_data.n_starts):
        ax.scatter(map_data.n_x*map_data.resolution_x*map_data.starts[i, 0], map_data.n_y*map_data.resolution_y*map_data.starts[i, 1], c='b')
        plt.text(map_data.n_x*map_data.resolution_x*map_data.starts[i, 0], map_data.n_y*map_data.resolution_y*map_data.starts[i, 1], 's')
    for i in range(map_data.n_tasks):
        ax.scatter(map_data.n_x*map_data.resolution_x*map_data.tasks[i, 0], map_data.n_y*map_data.resolution_y*map_data.tasks[i, 1], c='r')
        plt.text(map_data.n_x*map_data.resolution_x*map_data.tasks[i, 0], map_data.n_y*map_data.resolution_y*map_data.tasks[i, 1], 't')

    ax.plot([p[0] for p in path_robot_0], [p[1] for p in path_robot_0], 'r')
    plt.show()
            
  

if __name__ == "__main__":
    test_robot_2()