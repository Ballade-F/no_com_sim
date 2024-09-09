import numpy as np
import matplotlib.pyplot as plt
import map as mp
import ctrl
import path_planner
from task_allocation import hungarian
import time as TM

if __name__ == '__main__':
    n_starts = 4
    n_tasks = 4
    map = mp.Map(10, n_starts, n_tasks, 100, 100, 1, 1)
    map.setObstacleRandn(2025)
    # map.plot()
    # map.plotGrid()
    astar_planner = path_planner.AStarPlanner(map.grid_map)
    starts = map.starts_grid
    tasks = map.tasks_grid

    # calculate the distance matrix
    dist_matrix = np.zeros((n_starts, n_tasks))
    path_matrix = []
    for i in range(n_starts):
        for j in range(n_tasks):
            astar_planner.resetNodes()
            path, dist_matrix[i, j] = astar_planner.plan(starts[i], tasks[j])
            path_matrix.append(path)

    print(dist_matrix)

    # task allocation
    hungarian_solver = hungarian.Hungarian(dist_matrix)
    col_ind = hungarian_solver.get_col_ind()

    path_allot = []
    for i in range(n_starts):
        path_allot.append(path_matrix[i*n_tasks+col_ind[i]])

    #ctrl
    

    #plot path
    fig, ax = plt.subplots()
    ax.set_xlim(0, map.n_x)
    ax.set_ylim(0, map.n_y)
    img = 255-map.grid_map*255
    img = img.transpose()
    ax.imshow(img, cmap='gray')
    for ob_points in map.obstacles:
        ax.fill(ob_points[:, 0]*map.n_x, ob_points[:, 1]*map.n_y, 'r')
    # for path in path_matrix:
    #     ax.plot([x[0] for x in path], [x[1] for x in path], 'r')
    for path in path_allot:
        ax.plot([x[0] for x in path], [x[1] for x in path], 'r')

    map.plot()




