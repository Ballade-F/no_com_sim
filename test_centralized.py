import numpy as np
import matplotlib.pyplot as plt
import map as mp
import ctrl
import path_planner
from task_allocation import hungarian
import time as TM

if __name__ == '__main__':
    map = mp.Map(3, 2, 2, 100, 100, 1, 1)
    map.setObstacleRandn(2)
    map.plot()
    map.plotGrid()