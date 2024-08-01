import numpy as np
import matplotlib.pyplot as plt

#state space model
#state: [x, y, yaw, v, w]
class State():
    def __init__(self, x0:float=0, y0:float=0, yaw0:float=0, v0:float=0, w0:float=0, dt:float=0.1):
        self.X = np.array([x0, y0, yaw0, v0, w0])
        self.X_1 = np.array([x0, y0, yaw0, v0, w0])

        self.dt = dt

        self.n_states = 5
        self.n_observation = 2
        
    def update(self, v_new:float, w_new:float):
        self.X_1 = self.X
        self.X = np.array([self.X[0] + self.X[3] * np.cos(self.X[2]) * self.dt,
                           self.X[1] + self.X[3] * np.sin(self.X[2]) * self.dt,
                           self.X[2] + self.X[4] * self.dt,
                           v_new,
                           w_new])

    def observation(self):
        return self.X[:self.n_observation]
    
class ExtendKalman():
    def __init__(self, state:State):
        self.state = state
        
        self.P = np.eye(state.n_states)
        self.Q = np.eye(state.n_states)
        self.R = np.eye(state.n_observation)
        self.H = np.zeros((state.n_observation, state.n_states))
        self.I = np.eye(state.n_states)
        self.F = np.zeros((state.n_states, state.n_states))
        self.K = np.zeros((state.n_states, state.n_observation))
    
    def predict(self):
        self.F = np.array([[1, 0, -self.state.X_1[3] * np.sin(self.state.X_1[2]) * self.state.dt, np.cos(self.state.X_1[2]) * self.state.dt, 0],
                           [0, 1, self.state.X_1[3] * np.cos(self.state.X_1[2]) * self.state.dt, np.sin(self.state.X_1[2]) * self.state.dt, 0],
                           [0, 0, 1, 0, self.state.dt],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1]])
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z:np.ndarray):
        self.H = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0]])
        self.K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.state.X = self.state.X + self.K @ (z - self.H @ self.state.X)
        self.P = (self.I - self.K @ self.H) @ self.P


    