import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import utils.map as mp
import pnc.ctrl as ctrl
import pnc.path_planner as path_planner
from task_allocation import hungarian
import time as TM

if __name__ == '__main__':
    n_map = 50
    for seed in range(n_map):
        point_n_rng = np.random.default_rng(seed+n_map)
        n_points = point_n_rng.integers(5, 10)
        n_starts = n_points
        n_tasks = n_points
        map = mp.Map(0, n_starts, n_tasks, 100, 100, 1, 1)
        map.setObstacleRandn(seed)

        astar_planner = path_planner.AStarPlanner(map.grid_map, map.resolution_x, map.resolution_y)
        starts = map.starts_grid
        tasks = map.tasks_grid

        # calculate the distance matrix
        dist_matrix = np.zeros((n_starts, n_tasks))
        path_matrix = []
        for i in range(n_starts):
            for j in range(n_tasks):
                astar_planner.resetNodes()
                #path is index of grid nodes
                path, dist_matrix[i, j] = astar_planner.plan(starts[i], tasks[j])
                # path_true = [((p[0] + 0.5)/map.n_x, (p[1]+0.5)/map.n_y) for p in path]
                path_matrix.append(path)

        #task allocation
        hungarian_solver = hungarian.Hungarian(dist_matrix)
        col_ind = hungarian_solver.get_col_ind()
        path_allot_norm = []
        for i in range(n_starts):
            path_norm = [((p[0] + 0.5)/map.n_x, (p[1]+0.5)/map.n_y) for p in path_matrix[i*n_tasks+col_ind[i]]]
            path_allot_norm.append(path_norm)
            
        n = [len(path) for path in path_allot_norm]
        n_max = max(n)

        writeCsv_flag = True
        if writeCsv_flag:
            # Create directory for the map
            dir_name = f"intention_data/map_{seed}"
            os.makedirs(dir_name, exist_ok=True)
            with open(os.path.join(dir_name, "map_info.txt"), "w") as txt_file:
                txt_file.write(f"Map n_x: {map.n_x}\n")
                txt_file.write(f"Map n_y: {map.n_y}\n")
                txt_file.write(f"Map resolution_x: {map.resolution_x}\n")
                txt_file.write(f"Map resolution_y: {map.resolution_y}\n")
                txt_file.write(f"Number of starts: {n_starts}\n")
                txt_file.write(f"Number of tasks: {n_tasks}\n")

            with open(os.path.join(dir_name, "trajectory.csv"), "w", newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['Time','Type','ID','x', 'y','feature'])
                # simulate the trajectory
                for i in range(n_max+1):
                    for j in range(n_starts):
                        if i < n[j]:
                            writer.writerow([i, 'robot', j, path_allot_norm[j][i][0], path_allot_norm[j][i][1], col_ind[j]])
                        else:
                            if map.checkTaskFinish(path_allot_norm[j][-1][0], path_allot_norm[j][-1][1], col_ind[j], True):
                                col_ind[j] = -1
                            writer.writerow([i, 'robot', j, path_allot_norm[j][-1][0], path_allot_norm[j][-1][1], col_ind[j]])

                    for j in range(n_tasks):
                        writer.writerow([i, 'task', j, map.tasks[j,0], map.tasks[j,1], map.tasks_finish[j]])

        # plot the map
        if seed == n_map-1:
            fig, ax = plt.subplots()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            img = 255-map.grid_map*255
            img = img.transpose()
            ax.imshow(img, cmap='gray',vmin=0, vmax=255)
            for path in path_allot_norm:
                ax.plot([x[0] for x in path], [x[1] for x in path], 'r')

            map.plot()
            
