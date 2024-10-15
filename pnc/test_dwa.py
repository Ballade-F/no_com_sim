import unittest
import numpy as np
import matplotlib.pyplot as plt
from dwa import DWA

class TestDWAVisual(unittest.TestCase):
    def setUp(self):
        # 初始化一个简单的栅格地图
        grid_map = np.zeros((10, 10))
        grid_map[5, 5] = 1  # 添加一个障碍物

        # 初始化DWA对象
        self.dwa = DWA(
            v_ave=0.2,
            dt=0.1,
            predict_time=1.0,
            pos_factor=1.0,
            theta_factor=1.0,
            v_factor=1.0,
            w_factor=1.0,
            obstacle_factor=1.0,
            obstacle_r=0.5,
            resolution_x=0.1,
            resolution_y=0.1,
            grid_map=grid_map
        )

    def test_dwa_planner_visual(self):
        path = [(0, 0), (1, 1), (2, 2), (3, 3)]
        target_flag, v, w = self.dwa.DWA_Planner(path, 0, 0, 0, 0, 0)
        
        # # 可视化
        # fig, ax = plt.subplots()
        
        # # 绘制栅格地图
        # ax.imshow(self.dwa.grid_map.T, origin='lower', cmap='gray', alpha=0.3)
        
        # # 绘制路径
        # path_x, path_y = zip(*path)
        # ax.plot(path_x, path_y, 'g--', label='Path')
        
        # # 绘制初始位置
        # ax.plot(0, 0, 'bo', label='Start')
        
        # # 绘制目标位置
        # ax.plot(path[-1][0], path[-1][1], 'ro', label='Goal')
        
        # 绘制DWA规划的轨迹
        # state = [0, 0, 0, 0, 0]
        # traj, _ = self.dwa.calculate_traj(state, [v, w])
        # traj_x, traj_y = zip(*traj)
        # ax.plot(traj_x, traj_y, 'b-', label='DWA Trajectory')
        
        # ax.legend()
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('DWA Path Planning')
        # plt.grid(True)
        # plt.show()

if __name__ == '__main__':
    unittest.main()