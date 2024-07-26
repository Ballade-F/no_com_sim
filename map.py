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
        for ob_points in self.obstacles:
            ob_points[:, 0] = ob_points[:, 0] * self.n_x
            ob_points[:, 1] = ob_points[:, 1] * self.n_y
            



    def setObstacles(self, obstacles: list, start: list, tasks: list):
        self.obstacles = obstacles
        self.starts = start
        self.tasks = tasks

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

    def plot(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for ob_points in self.obstacles:
            ax.fill(ob_points[:, 0], ob_points[:, 1], 'k')
        ax.scatter(self.starts[:, 0], self.starts[:, 1], c='b')
        ax.scatter(self.tasks[:, 0], self.tasks[:, 1], c='r')
        plt.show()

if __name__ == '__main__':
    map = Map(3, 5, 5, 1, 1)
    map.setObstacleRandn(1)
    map.plot()



