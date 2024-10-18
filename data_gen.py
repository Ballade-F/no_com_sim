import csv
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import utils.map as mp
# import pnc.ctrl as ctrl
import pnc.path_planner as path_planner
from task_allocation import hungarian ,greedy_allocation_lib
import time as TM
from concurrent.futures import ProcessPoolExecutor


class AllocationDatasetGen():
    def __init__(self,dir,n_batch,batch_size,n_robot_min, n_robot_max, n_task_min, n_task_max, n_obstacle_min, n_obstacle_max, ob_points,seed=0, n_x=100, n_y=100, resolution_x=0.1, resolution_y=0.1,n_workers=4):
        self.dir = dir
        self.n_batch = n_batch
        self.batch_size = batch_size
        self.n_robot_min = n_robot_min
        self.n_robot_max = n_robot_max
        self.n_task_min = n_task_min
        self.n_task_max = n_task_max
        self.n_obstacle_min = n_obstacle_min
        self.n_obstacle_max = n_obstacle_max
        self.ob_points = ob_points
        self.seed = seed
        self.n_x = n_x
        self.n_y = n_y
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.n_workers = n_workers

    def allocationBatchGen(self,dir_name, rng:np.random.Generator , batch_size, n_robot, n_task, n_obstacle):
        # Save map information to a JSON file
        batch_info = {
            "batch_size": batch_size,
            "n_robot": n_robot,
            "n_task": n_task,
            "n_obstacle": n_obstacle,
            "ob_points": self.ob_points,
            "n_x": self.n_x,
            "n_y": self.n_y,
            "resolution_x": self.resolution_x,
            "resolution_y": self.resolution_y
        }
        with open(os.path.join(dir_name, "batch_info.json"), "w") as json_file:
            json.dump(batch_info, json_file, indent=4)

        for i in range(batch_size):
            dir_name_map = os.path.join(dir_name, f"map_{i}")
            os.makedirs(dir_name_map, exist_ok=True)
            map = mp.Map(n_obstacle, n_robot, n_task, self.n_x, self.n_y, self.resolution_x, self.resolution_y)
            map.setObstacleRandn(rng)
            astar_planner = path_planner.AStarPlanner(map.grid_map, map.resolution_x, map.resolution_y)
            starts = map.starts_grid
            tasks = map.tasks_grid
            points = np.concatenate((starts, tasks), axis=0)
            # 下三角矩阵
            costmat = np.full((n_robot+n_task, n_robot+n_task),fill_value=-1.0,dtype=float)
            for j in range(n_robot+n_task):
                for k in range(j+1):
                    astar_planner.resetNodes()
                    path, costmat[j, k] = astar_planner.plan(points[j], points[k])
                    # 对称矩阵
                    costmat[k, j] = costmat[j, k]

            # Save cost matrix to a npy file
            np.save(os.path.join(dir_name_map, "costmat.npy"), costmat)

            # Save map information to a csv file
            with open(os.path.join(dir_name_map, f"info.csv"), "w", newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Type", "x", "y"])
                for start in map.starts:
                    writer.writerow([-1, start[0], start[1]])
                for task in map.tasks:
                    writer.writerow([0, task[0], task[1]])
                for ob_idx, ob in enumerate(map.obstacles):
                    for point in ob:
                        writer.writerow([ob_idx+1, point[0], point[1]])

    def AllocationDatasetGen(self):
        os.makedirs(self.dir, exist_ok=True)
        dataset_info = {
            "time": TM.strftime("%Y-%m-%d %H:%M", TM.localtime()),
            "n_batch": self.n_batch,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "n_robot_min": self.n_robot_min,
            "n_robot_max": self.n_robot_max,
            "n_task_min": self.n_task_min,
            "n_task_max": self.n_task_max,
            "n_obstacle_min": self.n_obstacle_min,
            "n_obstacle_max": self.n_obstacle_max,
            "ob_points": self.ob_points,
            "n_x": self.n_x,
            "n_y": self.n_y,
            "resolution_x": self.resolution_x,
            "resolution_y": self.resolution_y
        }
        with open(os.path.join(self.dir, "dataset_info.json"), "w") as json_file:
            json.dump(dataset_info, json_file, indent=4)

        rng = np.random.default_rng(self.seed)

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for i in range(self.n_batch):
                dir_name = os.path.join(self.dir, f"batch_{i}")
                os.makedirs(dir_name, exist_ok=True)
                _n_robot = int(rng.integers(self.n_robot_min, self.n_robot_max + 1))
                _n_task = int(rng.integers(self.n_task_min, self.n_task_max + 1))
                _n_obstacle = int(rng.integers(self.n_obstacle_min, self.n_obstacle_max + 1))
                futures.append(executor.submit(self.allocationBatchGen, dir_name, rng, self.batch_size, _n_robot, _n_task, _n_obstacle))

            for future in futures:
                future.result()


class IntentionDatasetGen():
    def __init__(self,dir,n_scale,n_map,n_robot_min, n_robot_max, n_task_min, n_task_max, n_obstacle_min, n_obstacle_max, ob_points,seed=0, n_x=100, n_y=100, resolution_x=0.1, resolution_y=0.1, n_workers=4):
        self.dir = dir
        self.n_scale = n_scale
        self.n_map = n_map
        self.n_robot_min = n_robot_min
        self.n_robot_max = n_robot_max
        self.n_task_min = n_task_min
        self.n_task_max = n_task_max
        self.n_obstacle_min = n_obstacle_min
        self.n_obstacle_max = n_obstacle_max
        self.ob_points = ob_points
        self.seed = seed
        self.n_x = n_x
        self.n_y = n_y
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.n_workers = n_workers
    
    def intentionScaleGen(self,dir_name, rng:np.random.Generator, n_robot, n_task, n_obstacle):
        # save scale information to a JSON file
        scale_info = {
            "n_map": self.n_map,
            "n_robot": n_robot,
            "n_task": n_task,
            "n_obstacle": n_obstacle,
            "ob_points": self.ob_points,
            "n_x": self.n_x,
            "n_y": self.n_y,
            "resolution_x": self.resolution_x,
            "resolution_y": self.resolution_y
        }
        with open(os.path.join(dir_name, "scale_info.json"), "w") as json_file:
            json.dump(scale_info, json_file, indent=4)

        #对于每个map，需要一个csv存障碍，一个csv存机器人轨迹和目标完成情况
        for i_map in range(self.n_map):
            dir_name_map = os.path.join(dir_name, f"map_{i_map}")
            os.makedirs(dir_name_map, exist_ok=True)
            map = mp.Map(n_obstacle, n_robot, n_task, self.n_x, self.n_y, self.resolution_x, self.resolution_y)
            map.setObstacleRandn(rng)

            # #障碍csv
            # with open(os.path.join(dir_name_map, f"targets_obstacles.csv"), "w", newline='') as csv_file:
            #     writer = csv.writer(csv_file)
            #     writer.writerow(["Type", "x", "y"])
            #     # for task in map.tasks:
            #     #     writer.writerow([0, task[0], task[1]])
            #     for ob_idx, ob in enumerate(map.obstacles):
            #         for point in ob:
            #             writer.writerow([ob_idx+1, point[0], point[1]])
            
            # Save map information to a csv file
            with open(os.path.join(dir_name_map, f"info.csv"), "w", newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Type", "x", "y"])
                for start in map.starts:
                    writer.writerow([-1, start[0], start[1]])
                for task in map.tasks:
                    writer.writerow([0, task[0], task[1]])
                for ob_idx, ob in enumerate(map.obstacles):
                    for point in ob:
                        writer.writerow([ob_idx+1, point[0], point[1]])

            astar_planner = path_planner.AStarPlanner(map.grid_map, map.resolution_x, map.resolution_y)
            starts = map.starts_grid
            tasks = map.tasks_grid

            # calculate the distance matrix
            rt_matrix = np.zeros((n_robot, n_task))
            task_matrix = np.zeros((n_task, n_task))
            rt_path_norm = []
            task_path_norm = []
            for i in range(n_robot):
                for j in range(n_task):
                    astar_planner.resetNodes()
                    #path is index of grid nodes
                    path, rt_matrix[i, j] = astar_planner.plan(starts[i], tasks[j])
                    path_norm = [((p[0] + 0.5)/map.n_x, (p[1]+0.5)/map.n_y) for p in path]
                    rt_path_norm.append(path_norm)

            for i in range(n_task):
                for j in range(n_task):
                    astar_planner.resetNodes()
                    path, task_matrix[i, j] = astar_planner.plan(tasks[i], tasks[j])
                    path_norm = [((p[0] + 0.5)/map.n_x, (p[1]+0.5)/map.n_y) for p in path]
                    task_path_norm.append(path_norm)

            # allocation
            #TODO:留出接口，可以选择其他的分配方法
            allocation_solver = greedy_allocation_lib.GreedyTaskAllocationPlanner()
            robot_schedules = allocation_solver.greedy_allocate_mat(rt_matrix, task_matrix)

            path_allot_norm = [[] for _ in range(n_robot)] #每个机器人的路径
            paths_len = [[] for _ in range(n_robot)]#每段路径的长度
            for i in range(n_robot):
                path_robot = []
                path_num = []
                if len(robot_schedules[i]) == 0:
                    path_robot.append(map.starts[i])
                    path_num = [1]
                else:
                    path_robot = (rt_path_norm[i*n_task+robot_schedules[i][0]])#起点到第一个任务的路径
                    path_num.append(len(path_robot))
                    for j in range(1, len(robot_schedules[i])):
                        path_temp = task_path_norm[robot_schedules[i][j-1]*n_task+robot_schedules[i][j]]
                        path_robot += path_temp
                        path_num.append(len(path_temp))
                path_allot_norm[i] = path_robot
                paths_len[i] = path_num


            # Save robot trajectory and tasks to a csv file
            #TODO:换成dwa的轨迹
            n = [len(path) if len(robot_schedules[i]) != 0 else 0 for i, path in enumerate(path_allot_norm)] #如果没有任务，长度为0
            n_max = max(n)+1
            path_index = [0 for _ in range(n_robot)]
            points_num = [0 for _ in range(n_robot)]

            with open(os.path.join(dir_name_map, "trajectory.csv"), "w", newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['Time','Type','ID','x', 'y','feature'])
                # simulate the trajectory
                for i in range(n_max):
                    for j in range(n_robot):
                        if i < n[j]:
                            writer.writerow([i, 'robot', j, path_allot_norm[j][i][0], path_allot_norm[j][i][1], robot_schedules[j][path_index[j]]])
                            points_num[j] += 1
                            if points_num[j] >= paths_len[j][path_index[j]]:
                                map.checkTaskFinish(path_allot_norm[j][i][0], path_allot_norm[j][i][1], robot_schedules[j][path_index[j]], True)
                                path_index[j] += 1
                                points_num[j] = 0
                        else:
                            writer.writerow([i, 'robot', j, path_allot_norm[j][-1][0], path_allot_norm[j][-1][1], -1])

                    for j in range(n_task):
                        writer.writerow([i, 'task', j, map.tasks[j,0], map.tasks[j,1], int(map.tasks_finish[j])])
            
            # if i_map == 0:
            #     fig, ax = plt.subplots()
            #     ax.set_xlim(0, 1)
            #     ax.set_ylim(0, 1)
            #     img = 255-map.grid_map*255
            #     img = img.transpose()
            #     ax.imshow(img, cmap='gray',vmin=0, vmax=255)
            #     for path in path_allot_norm:
            #         ax.plot([x[0] for x in path], [x[1] for x in path], 'r')

            #     map.plot()      

    def IntentionDatasetGen(self):
        os.makedirs(self.dir, exist_ok=True)
        dataset_info = {
            "time": TM.strftime("%Y-%m-%d %H:%M", TM.localtime()),
            "n_scale": self.n_scale,
            "n_map": self.n_map,
            "seed": self.seed,
            "n_robot_min": self.n_robot_min,
            "n_robot_max": self.n_robot_max,
            "n_task_min": self.n_task_min,
            "n_task_max": self.n_task_max,
            "n_obstacle_min": self.n_obstacle_min,
            "n_obstacle_max": self.n_obstacle_max,
            "ob_points": self.ob_points,
            "n_x": self.n_x,
            "n_y": self.n_y,
            "resolution_x": self.resolution_x,
            "resolution_y": self.resolution_y
        }
        with open(os.path.join(self.dir, "dataset_info.json"), "w") as json_file:
            json.dump(dataset_info, json_file, indent=4)

        rng = np.random.default_rng(self.seed)

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for i in range(self.n_scale):
                dir_name = os.path.join(self.dir, f"scale_{i}")
                os.makedirs(dir_name, exist_ok=True)
                _n_robot = int(rng.integers(self.n_robot_min, self.n_robot_max + 1))
                _n_task = int(rng.integers(self.n_task_min, self.n_task_max + 1))
                _n_obstacle = int(rng.integers(self.n_obstacle_min, self.n_obstacle_max + 1))
                futures.append(executor.submit(self.intentionScaleGen, dir_name, rng, _n_robot, _n_task, _n_obstacle))

            for future in futures:
                future.result()
                print(f"Process {i+1}/{self.n_scale} completed")



if __name__ == '__main__':
    dir_allocation = "/home/data/wzr/no_com_1/data/allocation"
    dir_intention = "/home/data/wzr/no_com_1/data/intention"
    #intention n_batch
    n_scale = 128
    #intention n_map in each scale
    n_map = 16
    #allocation n_batch
    n_batch = 128
    #allocation
    batch_size = 64
    
    n_robot_min = 3
    n_robot_max = 8
    n_task_min = 10
    n_task_max = 30
    n_obstacle_min = 0
    n_obstacle_max = 8
    seed = 0
    n_x = 100
    n_y = 100
    resolution_x = 0.1
    resolution_y = 0.1
    ob_points = mp.n_ob_points
    n_workers = 64


    allocation_dataset_gen = AllocationDatasetGen(dir_allocation, n_batch, batch_size, n_robot_min, n_robot_max, n_task_min, n_task_max, 
                                                  n_obstacle_min, n_obstacle_max, ob_points, seed, n_x, n_y, resolution_x, resolution_y, n_workers)
    allocation_dataset_gen.AllocationDatasetGen()

    intention_dataset_gen = IntentionDatasetGen(dir_intention, n_scale, n_map, n_robot_min, n_robot_max, n_task_min, n_task_max,
                                                n_obstacle_min, n_obstacle_max, ob_points, seed, n_x, n_y, resolution_x, resolution_y, n_workers)
    intention_dataset_gen.IntentionDatasetGen()


    


