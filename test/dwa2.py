import numpy as np
from math import *
import matplotlib.pyplot as plt
 
#参数设置
V_Min = -0.5            #最小速度
V_Max = 3.0             #最大速度
W_Min = -50*pi/180.0    #最小角速度
W_Max = 50*pi/180.0     #最大角速度
Va = 0.5                #加速度
Wa = 30.0*pi/180.0      #角加速度
Vreso = 0.05            #速度分辨率
Wreso = 0.5*pi/180.0    #角速度分辨率
radius = 1              #机器人模型半径
Dt = 0.1                #时间间隔
Predict_Time = 4.0      #模拟轨迹的持续时间
alpha = 1.0             #距离目标点的评价函数的权重系数
Belta = 1.0             #速度评价函数的权重系数
Gamma = 1.0             #距离障碍物距离的评价函数的权重系数
 
#障碍物
Obstacle=np.array([[0,10],[2,10],[4,10],[6,10],[3,5],[4,5],[5,5],[6,5],[7,5],[8,5],[10,7],[10,9],[10,11],[10,13]])
#Obstacle = np.array([[0, 2]])
 
#距离目标点的评价函数
def Goal_Cost(Goal,Pos):
    return sqrt((Pos[-1,0]-Goal[0])**2+(Pos[-1,1]-Goal[1])**2)
 
#速度评价函数
def Velocity_Cost(Pos):
    return V_Max-Pos[-1,3]
 
#距离障碍物距离的评价函数
def Obstacle_Cost(Pos,Obstacle):
    MinDistance = float('Inf')          #初始化时候机器人周围无障碍物所以最小距离设为无穷
    for i in range(len(Pos)):           #对每一个位置点循环
        for j in range(len(Obstacle)):  #对每一个障碍物循环
            Current_Distance = sqrt((Pos[i,0]-Obstacle[j,0])**2+(Pos[i,1]-Obstacle[j,1])**2)  #求出每个点和每个障碍物距离
            if Current_Distance < radius:            #如果小于机器人自身的半径那肯定撞到障碍物了返回的评价值自然为无穷
                return float('Inf')
            if Current_Distance < MinDistance:
                MinDistance=Current_Distance         #得到点和障碍物距离的最小
 
    return 1/MinDistance
 
#速度采用
def V_Range(X):
    Vmin_Actual = X[3]-Va*Dt          #实际在dt时间间隔内的最小速度
    Vmax_actual = X[3]+Va*Dt          #实际载dt时间间隔内的最大速度
    Wmin_Actual = X[4]-Wa*Dt          #实际在dt时间间隔内的最小角速度
    Wmax_Actual = X[4]+Wa*Dt          #实际在dt时间间隔内的最大角速度
    VW = [max(V_Min,Vmin_Actual),min(V_Max,Vmax_actual),max(W_Min,Wmin_Actual),min(W_Max,Wmax_Actual)]  #因为前面本身定义了机器人最小最大速度所以这里取交集
    return VW
 
#一条模拟轨迹路线中的位置，速度计算
def Motion(X,u,dt):
    X[0]+=u[0]*dt*cos(X[2])           #x方向上位置
    X[1]+=u[0]*dt*sin(X[2])           #y方向上位置
    X[2]+=u[1]*dt                     #角度变换
    X[3]=u[0]                         #速度
    X[4]=u[1]                         #角速度
    return X
 
#一条模拟轨迹的完整计算
def Calculate_Traj(X,u):
    Traj=np.array(X)
    Xnew=np.array(X)
    time=0
    while time <=Predict_Time:        #一条模拟轨迹时间
        Xnew=Motion(Xnew,u,Dt)
        Traj=np.vstack((Traj,Xnew))   #一条完整模拟轨迹中所有信息集合成一个矩阵
        time=time+Dt
    return Traj
 
#DWA核心计算
def dwa_Core(X,u,goal,obstacles):
    vw=V_Range(X)
    best_traj=np.array(X)
    min_score=10000.0                 #随便设置一下初始的最小评价分数
    for v in np.arange(vw[0], vw[1], Vreso):         #对每一个线速度循环
        for w in np.arange(vw[2], vw[3], Wreso):     #对每一个角速度循环
            traj=Calculate_Traj(X,[v,w])
            goal_score=Goal_Cost(goal,traj)
            vel_score=Velocity_Cost(traj)
            obs_score=Obstacle_Cost(traj,Obstacle)
            score=goal_score+vel_score+obs_score
            if min_score>=score:                    #得出最优评分和轨迹
                min_score=score
                u=np.array([v,w])
                best_traj=traj
 
    return u,best_traj
 
x=np.array([2,2,45*pi/180,0,0])                          #设定初始位置，角速度，线速度
u=np.array([0,0])                                        #设定初始速度
goal=np.array([8,8])                                     #设定目标位置
global_tarj=np.array(x)
for i in range(1000):
    print(i)                                     #循环1000次，这里可以直接改成while的直到循环到目标位置
    u,current=dwa_Core(x,u,goal,Obstacle)
    x=Motion(x,u,Dt)
    global_tarj=np.vstack((global_tarj,x))                 #存储最优轨迹为一个矩阵形式每一行存放每一条最有轨迹的信息
    if sqrt((x[0]-goal[0])**2+(x[1]-goal[1])**2)<=radius:  #判断是否到达目标点
        print('Arrived')
        break
 
plt.plot(global_tarj[:,0],global_tarj[:,1],'*r',Obstacle[0:3,0],Obstacle[0:3,1],'-g',Obstacle[4:9,0],Obstacle[4:9,1],'-g',Obstacle[10:13,0],Obstacle[10:13,1],'-g')  #画出最优轨迹的路线
plt.show()