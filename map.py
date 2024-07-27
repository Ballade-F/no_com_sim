from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

#obstacle generator parameters
ob_r_factor = 0.5
ob_points_min = 3
ob_points_max = 10


class Map():
    def __init__(self, n_obstacles:int, n_starts:int, n_tasks:int, n_x:int, n_y:int, resolution_x:float, resolution_y:float):
        self.n_obstacles = n_obstacles
        self.n_starts = n_starts
        self.n_tasks = n_tasks
        self.n_x = n_x
        self.n_y = n_y
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.obstacles = []
        self.starts = []
        self.tasks = []
        self.grid_map = np.zeros((n_x, n_y))

    def _obstacle2grid(self):
        obstacles_back = deepcopy(self.obstacles)
        for ob_points in obstacles_back:
            ob_points[:, 0] = ob_points[:, 0] * self.n_x
            ob_points[:, 1] = ob_points[:, 1] * self.n_y

            min_x_index = int(np.min(ob_points[:, 0]))
            max_x_index = int(np.max(ob_points[:, 0]))
            min_y_index = int(np.min(ob_points[:, 1]))
            max_y_index = int(np.max(ob_points[:, 1]))

            # Create an edge table
            edge_table = []
            for i in range(len(ob_points)):
                x1, y1 = ob_points[i]
                x2, y2 = ob_points[(i + 1) % len(ob_points)]
                if y1 != y2:
                    #ensure x1, y1 is the lower endpoint
                    if y1 > y2:
                        x1, y1, x2, y2 = x2, y2, x1, y1
                    edge_table.append([y1, y2, x1, (x2 - x1) / (y2 - y1)])

            active_edge_table = []
            if min_y_index == max_y_index:
                self.grid_map[min_x_index:max_x_index + 1, min_y_index] = 1
                continue

            #if the x_axis ray crosses the polygon, the grids both up and down the ray should be filled
            for y in range(min_y_index + 1, max_y_index + 1):
                for edge in edge_table:
                    if edge[0] <= y and edge[1] > y:
                        active_edge_table.append(edge)
                # Sort active edge table by x-coordinate
                active_edge_table.sort(key=lambda edge: edge[2]+edge[3]*(y-edge[0]))
                # fill pixels between pairs of intersections
                for i in range(0, len(active_edge_table), 2):
                    x_start = int(active_edge_table[i][2]+active_edge_table[i][3]*(y-active_edge_table[i][0]))
                    x_end = int(active_edge_table[i + 1][2]+active_edge_table[i+1][3]*(y-active_edge_table[i+1][0]))
                    self.grid_map[x_start:x_end+1, y-1:y+1] = 1


            



    def setObstacles(self, obstacles: list, start: list, tasks: list):
        for ob_points in obstacles:
            ob_points[:, 0] = ob_points[:, 0] / (self.n_x*self.resolution_x)
            ob_points[:, 1] = ob_points[:, 1] / (self.n_y*self.resolution_y)
            self.obstacles.append(ob_points)

        start[:, 0] = start[:, 0] / (self.n_x*self.resolution_x)
        start[:, 1] = start[:, 1] / (self.n_y*self.resolution_y)
        self.starts = start

        tasks[:, 0] = tasks[:, 0] / (self.n_x*self.resolution_x)
        tasks[:, 1] = tasks[:, 1] / (self.n_y*self.resolution_y)
        self.tasks = tasks

        self._obstacle2grid()

    def setObstacleRandn(self, seed:int):
        rng = np.random.default_rng(seed)
        center_points = rng.uniform(0, 1, (self.n_obstacles, 2))
        ob_r_max = ob_r_factor/self.n_obstacles
        for i in range(self.n_obstacles):
            n_points = rng.integers(ob_points_min, ob_points_max)
            ob_angles = rng.uniform(0, 2*np.pi, n_points)
            ob_angles = np.sort(ob_angles)
            ob_r = rng.uniform(0, ob_r_max, n_points)

            ob_points = np.zeros((n_points, 2))
            ob_points[:, 0] = center_points[i, 0] + ob_r*np.cos(ob_angles)
            ob_points[:, 1] = center_points[i, 1] + ob_r*np.sin(ob_angles)

            ob_points = np.maximum(ob_points, 0)
            ob_points = np.minimum(ob_points, 1)
            
            self.obstacles.append(ob_points)

        self.starts = rng.uniform(0, 1, (self.n_starts, 2))
        self.tasks = rng.uniform(0, 1, (self.n_tasks, 2))

        self._obstacle2grid()

    def plot(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for ob_points in self.obstacles:
            ax.fill(ob_points[:, 0], ob_points[:, 1], 'k')
        ax.scatter(self.starts[:, 0], self.starts[:, 1], c='b')
        ax.scatter(self.tasks[:, 0], self.tasks[:, 1], c='r')
        plt.show()

    def plotGrid(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.n_x)
        ax.set_ylim(0, self.n_y)
        img = 255-self.grid_map*255
        ax.imshow(img, cmap='gray')
        plt.show()

if __name__ == '__main__':
    map = Map(1, 2, 2, 100, 100, 1, 1)
    map.setObstacleRandn(2)
    map.plot()
    map.plotGrid()



