import csv
import numpy as np
import matplotlib.pyplot as plt
import utils.map as mp
import pnc.ctrl as ctrl
import pnc.path_planner as path_planner
from task_allocation import hungarian
import time as TM

if __name__ == '__main__':
    n_starts = 4
    n_tasks = 4
    map = mp.Map(0, n_starts, n_tasks, 100, 100, 1, 1)
    map.setObstacleRandn(2026)
    # map.plot()
    # map.plotGrid()
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
            path_true = [((p[0] + 0.5)*map.resolution_x, (p[1]+0.5)*map.resolution_y) for p in path]
            path_matrix.append(path_true)

    print(dist_matrix)

    # task allocation
    hungarian_solver = hungarian.Hungarian(dist_matrix)
    col_ind = hungarian_solver.get_col_ind()

    path_allot = []
    for i in range(n_starts):
        path_allot.append(path_matrix[i*n_tasks+col_ind[i]])
    targets = col_ind

    #write path allot to csv
    n = [len(path) for path in path_allot]
    n_max = max(n)

    writeCsv_flag = False
    if writeCsv_flag:
        with open('path_allot.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time','Type','ID','x', 'y','Target/finish'])
            for i in range(n_max):
                for j in range(n_starts):
                    if i < n[j]:
                        writer.writerow([i, 'robot', j, path_allot[j][i][0], path_allot[j][i][1], targets[j]])
                    else:
                        writer.writerow([i, 'robot', j, path_allot[j][-1][0], path_allot[j][-1][1], targets[j]])
                        if map.checkTaskFinish(path_allot[j][-1][0], path_allot[j][-1][1], targets[j]):
                            targets[j] = -1
                for j in range(n_tasks):
                    writer.writerow([i, 'task', j, map.tasks[j,0], map.tasks[j,1], map.tasks_finish[j]])
     

    
 


    #plot path
    fig, ax = plt.subplots()
    ax.set_xlim(0, map.n_x)
    ax.set_ylim(0, map.n_y)
    img = 255-map.grid_map*255
    img = img.transpose()
    ax.imshow(img, cmap='gray',vmin=0, vmax=255)

    for ob_points in map.obstacles:
        ax.fill(ob_points[:, 0]*map.n_x, ob_points[:, 1]*map.n_y, 'r')
    # for path in path_matrix:
    #     ax.plot([x[0] for x in path], [x[1] for x in path], 'r')
    for path in path_allot:
        ax.plot([x[0] for x in path], [x[1] for x in path], 'r')

    map.plot()




