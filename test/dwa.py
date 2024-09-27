import math
import numpy as np
import matplotlib.pyplot as plt
 
 
class Config:
    # 速度限制
    v_max = 1.0
    v_min = 0.0
    w_max = 40 * math.pi / 180
    w_min = -40 * math.pi / 180
 
    # 加速度限制
    a_vmax = 0.1
    a_vmin = -0.1
    a_wmax = 40 * math.pi / 180
    a_wmin = - 40 * math.pi / 180
 
    # 采样间距
    v_sample = 0.01
    w_sample = 0.1 * math.pi / 180
 
    # 采样周期与预测时间
    dt = 0.1
    predict_time = 3.0
 
    # 评价函数权重
    alpha = 1.0
    beta = 2.0
    gamma = 2.0
 
    # 机器人半径和判断是否达到终点的判断距离
    robot_radius = 0.3
    judge_distance = 0.3
 
 
class DWA:
 
    def __init__(self, state: tuple[float, float, float, float, float]):
        """
        初始化机器人的状态
        """
        self.x = state[0]
        self.y = state[1]
        self.yaw = state[2]
        self.v = state[3]
        self.w = state[4]
 
    def run(self, goal: tuple[int, int], obstacles: list[tuple[int, int]]):
 
        window = self.getDynamicWindow()  # 速度采样
 
        predict_trajectories = self.getPredictTrajectories(window, obstacles)  # 轨迹预测
 
        best_trajectory = self.getBestTrajectory(predict_trajectories, goal, obstacles)  # 轨迹评价
 
        self.updateState((best_trajectory[0][3], best_trajectory[0][4]))  # 更新状态
 
        return predict_trajectories, best_trajectory
 
    def getDynamicWindow(self) -> list[tuple[float, float]]:
        """
        速度采样
        """
        # 速度限制
        V_vm = (Config.v_min, Config.v_max)
        V_wm = (Config.w_min, Config.w_max)
 
        # 加速度限制
        V_vd = (self.v + Config.a_vmin * Config.dt, self.v + Config.a_vmax * Config.dt)
        V_wd = (self.w + Config.a_wmin * Config.dt, self.w + Config.a_wmax * Config.dt)
 
        # 由以上两者决定的速度空间
        v_low = max(V_vm[0], V_vd[0])
        v_high = min(V_vm[1], V_vd[1])
        w_low = max(V_wm[0], V_wd[0])
        w_high = min(V_wm[1], V_wd[1])
 
        # 速度采样
        window = []
        n = int((v_high - v_low) / Config.v_sample)
        m = int((w_high - w_low) / Config.w_sample)
        for i in range(n):
            v = v_low + Config.v_sample * i
            for j in range(m):
                w = w_low + Config.w_sample * j
                window.append((v, w))
 
        return window
 
    def getPredictTrajectories(self, window: list[tuple[float, float]], obstacles) -> list[
        list[tuple[float, float, float, float, float]]]:
        """
        轨迹预测
        """
 
        predict_trajectories = []
        for v, w in window:
 
            predict_trajectory = []
            x = self.x
            y = self.y
            yaw = self.yaw
            predict_trajectory.append((x, y, yaw, v, w))
 
            for _ in range(int(Config.predict_time / Config.dt)):
                x += v * math.cos(yaw) * Config.dt
                y += v * math.sin(yaw) * Config.dt
                yaw += w * Config.dt
                predict_trajectory.append((x, y, yaw, v, w))
 
            # 判断该轨迹各点与最近障碍物的距离是否会导致碰撞，若会，则抛弃该轨迹
            if self.distFunc(predict_trajectory, obstacles) < Config.robot_radius:
                continue
            else:
                predict_trajectories.append(predict_trajectory)
 
        return predict_trajectories
 
    def getBestTrajectory(self,
                          trajectories: list[list[tuple[float, float, float, float, float]]],
                          goal: tuple[int, int],
                          obstacles: list[list]
                          ) -> list[tuple[float, float, float, float, float]]:
        """
        轨迹评价， 获得最佳轨迹
        """
 
        heads = []
        velocities = []
        dists = []
        for trajectory in trajectories:
            heads.append(self.headFunc(trajectory, goal))
            velocities.append(self.velocityFunc(trajectory))
            dists.append(self.distFunc(trajectory, obstacles))
 
        # 标准化
        norm_heads = [head / sum(heads) for head in heads]
        norm_velocities = [velocity / sum(velocities) for velocity in velocities]
        norm_dists = [dist / sum(dists) for dist in dists]
 
        Gs = []
        for norm_head, norm_velocity, norm_dist in zip(norm_heads, norm_velocities, norm_dists):
            G = Config.alpha * norm_head + Config.beta * norm_velocity + Config.gamma * norm_dist
            Gs.append(G)
 
        idx = Gs.index(max(Gs))
 
        return trajectories[idx]
 
    def updateState(self, control: tuple[float, float]):
        """
        根据控制指令更新状态
        """
 
        self.x += control[0] * math.cos(self.yaw) * Config.dt
        self.y += control[0] * math.sin(self.yaw) * Config.dt
        self.yaw += control[1] * Config.dt
        self.v = control[0]
        self.w = control[1]
 
    def headFunc(self, trajectory, goal):
        """
        方位角评价函数
        """
        dx = goal[0] - trajectory[-1][0]
        dy = goal[1] - trajectory[-1][1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1][2]
        cost = math.pi - abs(cost_angle)
 
        return cost
 
    @staticmethod
    def velocityFunc(trajectory):
        """
        速度评价函数
        """
        return trajectory[-1][3]
 
    @staticmethod
    def distFunc(trajectory: list[tuple[float, float, float, float, float]], obstacles: list[list]):
        """
        障碍物距离评价函数
        """
        min_dis = float("inf")
 
        for p in trajectory:
            for obs in obstacles:
                dis = math.hypot(p[0] - obs[0], p[1] - obs[1])
 
                if dis < min_dis:
                    min_dis = dis
 
        return min_dis
 
 
# 绘制机器人
def plot_robot(x, y, yaw):
    plt.arrow(x, y, math.cos(yaw), math.sin(yaw),
              head_length=0.1, head_width=0.1)
    plt.plot(x, y)
 
    circle = plt.Circle((x, y), Config.robot_radius, color="b")
    plt.gcf().gca().add_artist(circle)
    out_x, out_y = (np.array([x, y]) +
                    np.array([np.cos(yaw), np.sin(yaw)]) * Config.robot_radius)
    plt.plot([x, out_x], [y, out_y], "-k")
 
 
# 绘制轨迹
def plot_trajectories(trajectories: list[list[tuple[float, float, float, float, float]]], color='g'):
    for trajectory in trajectories:
        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]
 
        plt.plot(xs, ys, color=color)
 
 
if __name__ == '__main__':
    # 障碍物位置
    obstacles = \
        [[-1, -1],
         [0, 2],
         [4, 2],
         [5, 4],
         [5, 5],
         [5, 6],
         [5, 9],
         [8, 9],
         [7, 9],
         [8, 10],
         [9, 11],
         [12, 13],
         [12, 12],
         [15, 15],
         [13, 13]]
 
    # 目标点位置
    goal = (10, 10)
 
    # 机器人初始位置
    state = (0.0, 0.0, 0.0, 0.5, 0.0)
 
    # 算法初始化
    dwa = DWA(state)
 
    # 运行算法并绘图
    plt.ion()
    reality_trajectory_x = []  # 记录真实轨迹
    reality_trajectory_y = []
    while True:
        reality_trajectory_x.append(dwa.x)
        reality_trajectory_y.append(dwa.y)
 
        if math.hypot(dwa.x - goal[0], dwa.y - goal[1]) < Config.judge_distance:
            print("成功到达终点！")
            break
 
        plt.cla()
 
        # 绘制机器人、障碍物、目标
        plt.scatter(goal[0], goal[1])
        plt.scatter([obs[0] for obs in obstacles], [obs[1] for obs in obstacles])
        plot_robot(dwa.x, dwa.y, dwa.yaw)
 
        # 算法运行
        predict_trajectories, best_trajectory = dwa.run(goal, obstacles)
 
        # 绘制预测轨迹
        plot_trajectories(predict_trajectories)
        plot_trajectories([best_trajectory], color='r')
 
        # 绘制真实轨迹
        plt.plot(reality_trajectory_x, reality_trajectory_y, color='r')
 
        plt.pause(0.01)
 
    plt.ioff()
    plt.show()