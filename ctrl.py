import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import osqp
from qpsolvers import solve_qp

#state space model
#state: [x, y, yaw, v, w]
class State():
    def __init__(self, x0:float=0, y0:float=0, yaw0:float=0, v0:float=0, w0:float=0, dt:float=0.1,
                  noise_std_v:float=0, noise_std_w:float=0, noise_std_x:float=0, noise_std_y:float=0):
        self.X = np.array([x0, y0, yaw0, v0, w0])
        self.X_1 = np.array([x0, y0, yaw0, v0, w0])

        self.dt = dt

        self.n_states = 5
        self.n_observation = 2
        self.n_noise_pred = 2

        self.noise_std_v = noise_std_v
        self.noise_std_w = noise_std_w
        self.noise_std_x = noise_std_x
        self.noise_std_y = noise_std_y

        

    #with noise
    def forward(self):
        v_pred = self.X[3] + np.random.normal(0, self.noise_std_v)
        w_pred = self.X[4] + np.random.normal(0, self.noise_std_w)
        self.X_1 = self.X
        self.X = np.array([self.X_1[0] + v_pred*np.cos(self.X_1[2])*self.dt,
                            self.X_1[1] + v_pred*np.sin(self.X_1[2])*self.dt,
                            self.X_1[2] + w_pred*self.dt,
                            self.X_1[3],
                            self.X_1[4]])

    #without noise  
    def predict(self):
        self.X_1 = self.X
        self.X = np.array([self.X_1[0] + self.X_1[3]*np.cos(self.X_1[2])*self.dt,
                            self.X_1[1] + self.X_1[3]*np.sin(self.X_1[2])*self.dt,
                            self.X_1[2] + self.X_1[4]*self.dt,
                            self.X_1[3],
                            self.X_1[4]])
        
    def update_vw(self, v:float, w:float):
        self.X[3] = v
        self.X[4] = w
                           

    def observation(self):
        x_obs = self.X[0] + np.random.normal(0, self.noise_std_x)
        y_obs = self.X[1] + np.random.normal(0, self.noise_std_y)
        return np.array([x_obs, y_obs])
    
class ExtendKalman():
    def __init__(self, state:State,v_max:float=float('inf'), w_max:float=float('inf')):
        self.state = state
        self.Q = np.diag([state.noise_std_v**2, state.noise_std_w**2])
        self.R = np.diag([state.noise_std_x**2, state.noise_std_y**2])

        self.A = np.eye(state.n_states)
        self.W = np.zeros((state.n_states, state.n_noise_pred))
        
        self.P = np.eye(state.n_states)
        self.H = np.zeros((state.n_observation, state.n_states))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.K = np.zeros((state.n_states, state.n_observation))

        self.v_max = v_max
        self.w_max = w_max

    def predict(self):
        self.A[0, 2] = -self.state.X[3]*np.sin(self.state.X[2])*self.state.dt
        self.A[0, 3] = np.cos(self.state.X[2])*self.state.dt
        self.A[1, 2] = self.state.X[3]*np.cos(self.state.X[2])*self.state.dt
        self.A[1, 3] = np.sin(self.state.X[2])*self.state.dt
        self.A[2, 4] = self.state.dt

        self.W[0, 0] = 0.5*self.state.dt**2*np.cos(self.state.X[2])
        self.W[1, 0] = 0.5*self.state.dt**2*np.sin(self.state.X[2])
        self.W[2, 1] = 0.5*self.state.dt**2
        self.W[3, 0] = self.state.dt
        self.W[4, 1] = self.state.dt

        self.P = np.dot(self.A, self.P).dot(self.A.T) + np.dot(self.W, self.Q).dot(self.W.T)

        self.state.predict()

        return self.state.X

    #z: observation [x_obs, y_obs]
    #Posterior estimation of state
    def update(self, z:np.array):
        self.K = np.dot(self.P, self.H.T).dot(np.linalg.inv(np.dot(self.H, self.P).dot(self.H.T) + self.R))
        self.state.X = self.state.X + np.dot(self.K, z - self.state.observation())
        self.P = np.dot(np.eye(self.state.n_states) - np.dot(self.K, self.H), self.P)

        #limit v, w estimation
        self.state.X[3] = min(self.v_max, max(0, self.state.X[3]))
        self.state.X[4] = min(self.w_max, max(-self.w_max, self.state.X[4]))
        
        return self.state.X

#x: [x, y, yaw], u: [v, w], used for MPC
class CarModel():
    def __init__(self, x0:float=0, y0:float=0, yaw0:float=0, v0:float=0, w0:float=0, dt:float=0.1):
        self.x = x0
        self.y = y0
        self.yaw = yaw0
        self.v = v0
        self.w = w0

        self.dt = dt

        self.n_x = 3
        self.n_u = 2
        #asmatrix : Shallow copy
        self.state = np.asmatrix([self.x, self.y, self.yaw]).T

    #v at t, w at t, update x, y, yaw at t+1
    def update(self,v:float, w:float):
        self.x = self.x + v*np.cos(self.yaw)*self.dt
        self.y = self.y + v*np.sin(self.yaw)*self.dt
        self.yaw = self.yaw + w*self.dt
        self.v = v
        self.w = w
        
    # return linearized and discretized state matrix A and B at state_ref
    def stateSpaceModel(self, state_ref:np.array, u_ref:np.array):
        x_ref = state_ref[0]
        y_ref = state_ref[1]
        yaw_ref = state_ref[2]
        v_ref = u_ref[0]
        w_ref = u_ref[1]

        A_hat = np.asmatrix([[1, 0, -v_ref*np.sin(yaw_ref)*self.dt],
                         [0, 1, v_ref*np.cos(yaw_ref)*self.dt],
                         [0, 0, 1]])
        B_hat = np.asmatrix([[np.cos(yaw_ref)*self.dt, 0],
                         [np.sin(yaw_ref)*self.dt, 0],
                         [0, self.dt]])
        return A_hat, B_hat
        
        

#x: [x, y, yaw], u: [v, w]
class MPCCtrl():
    def __init__(self, A_hat, B_hat, Q, R, Qf, N = 10):
        self.A_hat = A_hat
        self.B_hat = B_hat
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.N = N    # prediction horizon
        self.nx = B_hat.shape[0]
        self.nu = B_hat.shape[1]
        
    # optimize state from k+1 to k+N, u from k to k+N-1
    def solve(self, error_state0, A_hat, B_hat, Q, R, Qf, N = 10):
        self.N = N
        self.nx = B_hat.shape[0]
        self.nu = B_hat.shape[1]
        self.A_hat = A_hat
        self.B_hat = B_hat
        self.Q = Q
        self.R = R
        self.Qf = Qf

        A_power = [np.linalg.matrix_power(A_hat, i) for i in range(N+1)]

        # X = Mx_k + CU
        M = np.zeros((self.N*self.nx, self.nx))
        C = np.zeros((self.N*self.nx, self.N*self.nu))

        for i in range(N):
            M[i*self.nx:(i+1)*self.nx, :] = A_power[i+1]
            for j in range(i):
                C[i*self.nx:(i+1)*self.nx, j*self.nu:(j+1)*self.nu] = A_power[i-j]*B_hat

        Q_bar = np.kron(np.eye(self.N + 1), Q)
        Q_bar[self.N * self.nx : (1 + self.N) * self.nx, self.N * self.nx : (1 + self.N) * self.nx:] = Qf
        R_bar = np.kron(np.eye(self.N), R)

        #J = x.T * Q_bar * x + u.T * R_bar * u
        #J = (M*x_k + C*u).T * Q_bar * (M*x_k + C*u) + u.T * R_bar * u
        #J = u.T * (C.T * Q_bar * C + R_bar) * u + 2 * x_k.T * M.T * Q_bar * C * u + (x_k.T * M.T * Q_bar * M * x_k)
        E = M.T * Q_bar * C
        # min 0.5x^T P x + q^T x
        P = 2 * (C.T * Q_bar * C + R_bar)
        q = 2 * E * error_state0

        #solve u
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