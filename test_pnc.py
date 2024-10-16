
import numpy as np
import matplotlib.pyplot as plt
import utils.map as map
import pnc.path_planner as path_planner
import pnc.dwa as dwa

def test_dwa():
    #map
    map_data = map.Map(1,1,1,100,100,0.1,0.1)
    obstacle = []
    ob_0 = np.array([[4.0, 6.0], [6.0, 6.0],[6.0, 4.0],[4.0, 4.0]])
    obstacle.append(ob_0)
    start = np.array([[1.0, 1.0]])
    task = np.array([[9.0, 9.0]])
    map_data.setObstacles(obstacle,start,task)
    # map_data.plot()
    # map_data.plotGrid()

    #path_planner
    planner = path_planner.AStarPlanner(map_data.grid_map, map_data.resolution_x, map_data.resolution_y)
    starts = map_data.starts_grid
    tasks = map_data.tasks_grid
    path_index = planner.plan(starts[0], tasks[0])
    path_true = [((p[0]+0.5)*map_data.resolution_x, (p[1]+0.5)*map_data.resolution_y) for p in path_index]


if __name__ == '__main__':
    test_dwa()