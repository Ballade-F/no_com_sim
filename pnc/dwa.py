import numpy as np
from math import *
import matplotlib.pyplot as plt

class DWA:
    def __init__(self, V_Min=-0.5, V_Max=3.0, W_Min=-50*pi/180.0, W_Max=50*pi/180.0, Va=0.5, Wa=30.0*pi/180.0, Vreso=0.05, Wreso=0.5*pi/180.0, radius=1, Dt=0.1, Predict_Time=4.0, alpha=1.0, Belta=1.0, Gamma=1.0):
        self.V_Min = V_Min
        self.V_Max = V_Max
        self.W_Min = W_Min
        self.W_Max = W_Max
        self.Va = Va
        self.Wa = Wa
        self.Vreso = Vreso
        self.Wreso = Wreso
        self.radius = radius
        self.Dt = Dt
        self.Predict_Time = Predict_Time
        self.alpha = alpha
        self.Belta = Belta
        self.Gamma = Gamma

    def Goal_Cost(self, Goal, Pos):
        return sqrt((Pos[-1, 0] - Goal[0]) ** 2 + (Pos[-1, 1] - Goal[1]) ** 2)

    def Velocity_Cost(self, Pos):
        return self.V_Max - Pos[-1, 3]

    def Obstacle_Cost(self, Pos, Obstacle):
        MinDistance = float('Inf')
        for i in range(len(Pos)):
            for j in range(len(Obstacle)):
                Current_Distance = sqrt((Pos[i, 0] - Obstacle[j, 0]) ** 2 + (Pos[i, 1] - Obstacle[j, 1]) ** 2)
                if Current_Distance < self.radius:
                    return float('Inf')
                if Current_Distance < MinDistance:
                    MinDistance = Current_Distance

        return 1 / MinDistance

    