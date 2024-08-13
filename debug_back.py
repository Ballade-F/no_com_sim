import math
import osqp
import numpy as np
from scipy import sparse
from qpsolvers import solve_qp
# from celluloid import Camera # 保存动图时用，pip install celluloid
from matplotlib import pyplot as plt

class MyReferencePath:
    def __init__(self):
        # set reference trajectory
        # refer_path包括4维：位置x, 位置y， 轨迹点的切线方向, 曲率k 
        self.refer_path = np.zeros((1000, 4))
        self.refer_path[:,0] = np.linspace(0, 50, 1000) # x
        self.refer_path[:,1] = 2*np.sin(self.refer_path[:,0]/3.0)+2.5*np.cos(self.refer_path[:,0]/2.0) # y
        # 使用差分的方式计算路径点的一阶导和二阶导，从而得到切线方向和曲率
        for i in range(len(self.refer_path)):
            if i == 0:
                dx = self.refer_path[i+1,0] - self.refer_path[i,0]
                dy = self.refer_path[i+1,1] - self.refer_path[i,1]
                ddx = self.refer_path[2,0] + self.refer_path[0,0] - 2*self.refer_path[1,0]
                ddy = self.refer_path[2,1] + self.refer_path[0,1] - 2*self.refer_path[1,1]
            elif i == (len(self.refer_path)-1):
                dx = self.refer_path[i,0] - self.refer_path[i-1,0]
                dy = self.refer_path[i,1] - self.refer_path[i-1,1]
                ddx = self.refer_path[i,0] + self.refer_path[i-2,0] - 2*self.refer_path[i-1,0]
                ddy = self.refer_path[i,1] + self.refer_path[i-2,1] - 2*self.refer_path[i-1,1]
            else:
                dx = self.refer_path[i+1,0] - self.refer_path[i,0]
                dy = self.refer_path[i+1,1] - self.refer_path[i,1]
                ddx = self.refer_path[i+1,0] + self.refer_path[i-1,0] - 2*self.refer_path[i,0]
                ddy = self.refer_path[i+1,1] + self.refer_path[i-1,1] - 2*self.refer_path[i,1]
            self.refer_path[i,2]=math.atan2(dy,dx) # yaw
            # 计算曲率:设曲线r(t) =(x(t),y(t)),则曲率k=(x'y" - x"y')/((x')^2 + (y')^2)^(3/2).
            # 参考：https://blog.csdn.net/weixin_46627433/article/details/123403726
            self.refer_path[i,3]=(ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2)) # 曲率k计算
            
    def calc_track_error(self, x, y):
        """计算跟踪误差

        Args:
            x (_type_): 当前车辆的位置x
            y (_type_): 当前车辆的位置y

        Returns:
            _type_: _description_
        """
        # 寻找参考轨迹最近目标点
        d_x = [self.refer_path[i,0]-x for i in range(len(self.refer_path))] 
        d_y = [self.refer_path[i,1]-y for i in range(len(self.refer_path))] 
        d = [np.sqrt(d_x[i]**2+d_y[i]**2) for i in range(len(d_x))]
        s = np.argmin(d) # 最近目标点索引

        return s

class CarModel():
    def __init__(self, x0:float=0, y0:float=0, yaw0:float=0, v0:float=0, w0:float=0, dt:float=0.1):
        self.x = x0
        self.y = y0
        self.yaw = yaw0
        self.v = v0
        self.w = w0

        self.dt = dt

        self.nx = 3
        self.nu = 2
        #asmatrix : Shallow copy, shape = (3, 1)
        self.state = np.asmatrix([self.x, self.y, self.yaw]).T
        # self.u = np.asmatrix([self.v, self.w]).T

    #v at t, w at t, update x, y, yaw at t+1
    def update(self, w:float):
        v = self.v
        self.x = self.x + v*np.cos(self.yaw)*self.dt
        self.y = self.y + v*np.sin(self.yaw)*self.dt
        self.yaw = self.yaw + w*self.dt
        self.v = v
        self.w = w
        self.state = np.asmatrix([self.x, self.y, self.yaw]).T
        # self.u = np.asmatrix([self.v, self.w]).T
        
    # return linearized and discretized state matrix A and B at state_ref
    def stateSpaceModel(self):
        yaw_ref = self.yaw
        v_ref = self.v
        A_hat = np.asmatrix([[1, 0, -v_ref*np.sin(yaw_ref)*self.dt],
                         [0, 1, v_ref*np.cos(yaw_ref)*self.dt],
                         [0, 0, 1]])
        B_hat = np.asmatrix([[np.cos(yaw_ref)*self.dt, 0],
                         [np.sin(yaw_ref)*self.dt, 0],
                         [0, self.dt]])
        return A_hat, B_hat
        
        


class MPC:
    def __init__(self, Ad, Bd, Q, R, Qf, N = 10):
        self.Ad = Ad
        self.Bd = Bd
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.N = N    # 预测步数
        self.nx = Bd.shape[0]
        self.nu = Bd.shape[1]

    def solve(self, x0, Ad, Bd, Q, R, Qf, N = 10):
        self.Ad = Ad
        self.Bd = Bd
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.N = N    # 预测步数
        self.nx = Bd.shape[0]
        self.nu = Bd.shape[1]

        A_powers = []
        for i in range(self.N + 1):
            A_powers.append(np.linalg.matrix_power(Ad, i))

        C = np.zeros(((self.N + 1) * self.nx, self.N * self.nu))
        M = np.zeros(((self.N + 1) * self.nx, self.nx))
        for i in range(self.N + 1):
            for j in range(self.N):
                if i - j - 1 >= 0:
                    C_ij = A_powers[i - j - 1] * self.Bd
                    C[i * self.nx : (i + 1) * self.nx, j * self.nu : (j + 1) * self.nu] = C_ij
                else:
                    C_ij = np.zeros((self.nx, self.nu))
                    C[i * self.nx : (i + 1) * self.nx, j * self.nu : (j + 1) * self.nu] = C_ij
            M[i * self.nx : (i + 1) * self.nx, :] = A_powers[i]

        Q_bar = np.kron(np.eye(self.N + 1), Q)
        Q_bar[self.N * self.nx : (1 + self.N) * self.nx, self.N * self.nx : (1 + self.N) * self.nx:] = Qf
        R_bar = np.kron(np.eye(self.N), R)
        E = M.T * Q_bar * C

        P = 2 * (C.T * Q_bar * C + R_bar)
        q = 2 * E.T * x0

        # Gx <= h
        G_ = np.eye(self.N * self.nu)
        G = np.block([                   # 不等式约束矩阵
            [G_, np.zeros_like(G_)],
            [np.zeros_like(G_), -G_]
        ])
        h = np.vstack(np.ones((2 * self.N * self.nu, 1)) * 999) # 不等式约束向量

        # Ax = b
        A = None # 等式约束矩阵
        b = None # 等式约束向量

        # 转换为稀疏矩阵的形式能加速计算
        P = sparse.csc_matrix(P)
        q = np.asarray(q)
        if G is None:
            pass
        else:
            G = sparse.csc_matrix(G)
        if A is None:
            pass
        else:
            A = sparse.csc_matrix(A)

        res = solve_qp(P, q, G, h, A, b, solver="osqp")

        return res
    

# 是否保存动图
IS_SAVE_GIF = False

def vehicle_mpc_main():
    # 时间范围
    T = 100
    dt = 0.1
    num_steps = int(T / dt)
    time = np.linspace(0, T, num_steps + 1).T

    reference_path = MyReferencePath()
    goal = reference_path.refer_path[-1, 0:2]
    trajectory_x = []
    trajectory_y = []
    fig = plt.figure(1)

    vehicle = CarModel(v0=2.0)
    Q = np.matrix(np.eye(vehicle.nx) * 1)
    R = np.matrix(np.eye(vehicle.nu) * 0.5)
    Qf = np.matrix(np.eye(vehicle.nx) * 1)
    mpc = MPC(np.eye(vehicle.nx), np.eye(vehicle.nu), Q, R, Qf, N = 10)

    # if True == IS_SAVE_GIF:
    #     camera = Camera(fig)

    for i in range(num_steps):
        # 参考线轨迹部分
        s0 = reference_path.calc_track_error(vehicle.x, vehicle.y)
        # delta_ref = math.atan2(vehicle.L * k, 1)

        # 数学模型更新，相当于建模
        Ad, Bd = vehicle.stateSpaceModel()
        nx = Ad.shape[0]
        nu = Bd.shape[1]
        # Ad = np.eye(nx) + A * dt
        # Bd = B * dt

        # 更新MPC控制器的系统矩阵并求解最优控制量
        u_list = mpc.solve((vehicle.state - reference_path.refer_path[s0, 0:3].reshape(-1, 1)), Ad, Bd, Q, R, Qf, 10)
        u = np.matrix(u_list[0 : nu]).T
        # u[1, 0] = u[1, 0] + delta_ref

        # MPC控制下的系统状态更新
        vehicle.update( u[1, 0])

        # 显示动图
        trajectory_x.append(vehicle.x)
        trajectory_y.append(vehicle.y)

        if False == IS_SAVE_GIF:
            plt.cla()
        plt.gca().set_aspect('equal', adjustable='box')
        # vehicle.draw_vehicle(plt)
        plt.plot(reference_path.refer_path[:, 0], reference_path.refer_path[:, 1], "-.b",  linewidth=1.0, label="course") # 参考线轨迹
        plt.plot(trajectory_x, trajectory_y, 'red') # 车辆轨迹
        plt.xlim(-5, 55)
        plt.ylim(-10, 20)
        if False == IS_SAVE_GIF:
            plt.pause(0.001)

        # if True == IS_SAVE_GIF:
        #     camera.snap()

        # 判断是否到达最后一个点
        if np.linalg.norm([vehicle.x, vehicle.y] - goal) <= 0.5:
            print("reach goal")
            break
    

if __name__ == "__main__":

    # 一个二阶不稳定系统的MPC控制演示demo
    #mpc_main()

    # 以车辆后轴中心为中心的车辆运动学模型的MPC控制演示demo
    vehicle_mpc_main()