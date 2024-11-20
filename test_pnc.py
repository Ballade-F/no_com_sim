import time as TM
import numpy as np
import matplotlib.pyplot as plt
import utils.map as map
import pnc.path_planner as path_planner
import pnc.dwa as dwa
import pnc.ctrl as ctrl

from math import sin, cos


def _state_update(state,u,dt):
    state[0] += u[0]*cos(state[2])*dt
    state[1] += u[0]*sin(state[2])*dt
    state[2] += u[1]*dt
    state[3] = u[0]
    state[4] = u[1]
    return state

def _judge_arrival(state, target, r):
    if np.linalg.norm(state[:2]-target[:2]) < r:
        print('arrived')
        return True
    else:
        return False

def test_dwa():
    #map
    map_data = map.Map(1,1,1,100,100,0.1,0.1)
    obstacle = []
    ob_0 = np.array([[4.0, 6.0], [6.0, 6.0],[6.0, 4.0],[4.0, 4.0]])
    obstacle.append(ob_0)
    start = np.array([[1.0, 1.0]])
    task = np.array([[9.0, 9.0]])
    map_data.setObstacles(obstacle,start,task)
    # map_data.plot()
    # map_data.plotGrid()

    #path_planner
    planner = path_planner.AStarPlanner(map_data.grid_map, map_data.resolution_x, map_data.resolution_y)
    starts = map_data.starts_grid
    tasks = map_data.tasks_grid
    path_index, cost= planner.plan(starts[0], tasks[0])
    path_true = [((p[0]+0.5)*map_data.resolution_x, (p[1]+0.5)*map_data.resolution_y) for p in path_index]

    #dwa
    x = start[0,0]
    y = start[0,1]
    theta = 3.14
    v = 0.0
    w = 0.0
    state_robot = np.array([x, y, theta, v, w])
    u_state = np.array([0.0, 0.0])
    target = task[0]

    v_ave = 0.2
    dt = 0.1
    predict_time = 2.5
    pos_factor = 100
    theta_factor = 10
    # 最后一个点的代价权重 与 （其他轨迹点之平均 的权重） 的比例
    final_factor = 0.05
    v_factor = 40
    w_factor = 20
    obstacle_factor = 20
    obstacle_r = 0.4
    resolution_x = map_data.resolution_x
    resolution_y = map_data.resolution_y
    grid_map = map_data.grid_map

    dwa_planner = dwa.DWA(v_ave, dt, predict_time, pos_factor, theta_factor, v_factor, w_factor, obstacle_factor,final_factor,
                           obstacle_r, resolution_x, resolution_y, grid_map,False,n_workers=4)
    counter = 0
    path_dwa = []
    plot_u = []
    path_dwa.append((state_robot[0], state_robot[1]))
    plot_u.append((counter, u_state[0], u_state[1]))
    arrival_flag = False
    
    while not arrival_flag:
        counter += 1
        time_1 = TM.time()
        target_flag, u_state[0], u_state[1] = dwa_planner.DWA_Planner(path_true, state_robot)
        time_2 = TM.time()
        print('time: ', time_2 - time_1)

        if not target_flag:
            print('dwa failed')
            break
        state_robot = _state_update(state_robot, u_state, dt)
        path_dwa.append((state_robot[0], state_robot[1]))
        plot_u.append((counter, u_state[0], u_state[1]))
        arrival_flag = _judge_arrival(state_robot, target, 0.05)

        if counter > 1000:
            print('dwa failed')
            break

    #plot path
    fig, ax = plt.subplots()
    ax.set_xlim(0, map_data.n_x*map_data.resolution_x)
    ax.set_ylim(0, map_data.n_y*map_data.resolution_y)
    for ob_points in map_data.obstacles:
        ax.fill(map_data.n_x*map_data.resolution_x*ob_points[:, 0], map_data.n_y*map_data.resolution_y*ob_points[:, 1], 'g')
    for i in range(map_data.n_starts):
        ax.scatter(map_data.n_x*map_data.resolution_x*map_data.starts[i, 0], map_data.n_y*map_data.resolution_y*map_data.starts[i, 1], c='b')
        plt.text(map_data.n_x*map_data.resolution_x*map_data.starts[i, 0], map_data.n_y*map_data.resolution_y*map_data.starts[i, 1], 's')
    for i in range(map_data.n_tasks):
        ax.scatter(map_data.n_x*map_data.resolution_x*map_data.tasks[i, 0], map_data.n_y*map_data.resolution_y*map_data.tasks[i, 1], c='r')
        plt.text(map_data.n_x*map_data.resolution_x*map_data.tasks[i, 0], map_data.n_y*map_data.resolution_y*map_data.tasks[i, 1], 't')

    ax.plot([x[0] for x in path_true], [x[1] for x in path_true], 'r')
    ax.plot([x[0] for x in path_dwa], [x[1] for x in path_dwa], 'b')
    plt.show()
    #plot u
    fig, ax = plt.subplots()
    ax.plot([x[0] for x in plot_u], [x[1] for x in plot_u], 'r', label='v')
    ax.plot([x[0] for x in plot_u], [x[2] for x in plot_u], 'b', label='w')
    plt.legend()
    plt.show()


def test_mpc():
    #generate random path
    v = 0.4
    w = 0.25
    dt = 0.1
    time = 30
    n = int(time/dt)
    x = np.zeros(n)
    y = np.zeros(n)
    yaw = np.zeros(n)
    car_state = ctrl.State(x0=0, y0=0, yaw0=0, v0=v, w0=w, dt=dt)

    for i in range(n):
        x[i] = car_state.X[0]
        y[i] = car_state.X[1]
        yaw[i] = car_state.X[2]
        car_state.forward()
        if i == 100:
            car_state.update_vw(0.4, -0.15)
        if i == 200:
            car_state.update_vw(0.4, 0.15)

    # plt.plot(x, y)
    # plt.show()

    #test EKF
    noise_std_v = 0.1
    noise_std_w = 0.1
    noise_std_x = 0.05
    noise_std_y = 0.05
    car_state_noise = ctrl.State(x0=0, y0=0, yaw0=0, v0=v, w0=w, dt=dt,
                                 noise_std_v=noise_std_v, noise_std_w=noise_std_w,
                                 noise_std_x=noise_std_x, noise_std_y=noise_std_y)
    x_noise = np.zeros(n)
    y_noise = np.zeros(n)
    x_noise[0] = car_state_noise.X[0]
    y_noise[0] = car_state_noise.X[1]
    car_state_ekf = ctrl.State(x0=0, y0=0, yaw0=0, v0=0, w0=0, dt=dt,
                           noise_std_v=noise_std_v, noise_std_w=noise_std_w,
                           noise_std_x=noise_std_x, noise_std_y=noise_std_y)
    ekf = ctrl.ExtendKalman(car_state_ekf)
    x_ekf = np.zeros(n)
    y_ekf = np.zeros(n)
    x_ekf[0] = car_state_ekf.X[0]
    y_ekf[0] = car_state_ekf.X[1]
    x_observation = np.zeros(n)
    y_observation = np.zeros(n)
    x_observation[0] = car_state_noise.X[0]
    y_observation[0] = car_state_noise.X[1]
    for i in range(1, n):
        x_noise[i] = car_state_noise.X[0]
        y_noise[i] = car_state_noise.X[1]
        car_state_noise.forward()
        z = car_state_noise.observation()
        ekf.predict()
        ekf.update(z)
        x_observation[i] = z[0]
        y_observation[i] = z[1]
        x_ekf[i] = car_state_ekf.X[0]
        y_ekf[i] = car_state_ekf.X[1]

        if i == 100:
            car_state_noise.update_vw(0.4, -0.15)
        if i == 200:
            car_state_noise.update_vw(0.4, 0.25)
    
    plt.plot(x_noise, y_noise, label='noisy')
    plt.plot(x_ekf, y_ekf, label='ekf')
    plt.plot(x_observation, y_observation, label='observation')
    plt.legend()
    plt.show()

    #test MPC
    vehicle = ctrl.CarModel(yaw0=1.5)
    Q = np.matrix(np.eye(vehicle.n_x) * 4)
    R = np.matrix(np.eye(vehicle.n_u) * 1)
    Qf = np.matrix(np.eye(vehicle.n_x) * 4)
    A_hat, B_hat = vehicle.stateSpaceModel(vehicle.state, vehicle.u)
    N_mpc = 10
    mpc = ctrl.MPCCtrl(A_hat, B_hat, Q, R, Qf, N = N_mpc)

    x_mpc = np.zeros(n)
    y_mpc = np.zeros(n)

    track_ref = np.zeros((n, vehicle.n_x))
    track_ref[:, 0] = x
    track_ref[:, 1] = y
    track_ref[:, 2] = yaw
    state_ref = np.zeros((N_mpc, vehicle.n_x))
    state_ref_0 = np.zeros(vehicle.n_x)
    
    time_1 = 0
    time_2 = 0
    for i in range(n):
        x_mpc[i] = vehicle.x
        y_mpc[i] = vehicle.y

        #debug
        # print('state: ', vehicle.state)
        # print('u: ', vehicle.u)

        time_1 = TM.time()

        A_hat, B_hat = vehicle.stateSpaceModel(vehicle.state, vehicle.u)

        #debug
        # print('A_hat:\n', A_hat, '\n B_hat: \n', B_hat)

        idx = mpc.calc_track_idx(vehicle.x, vehicle.y, track_ref)
        state_error0 = vehicle.state - track_ref[idx].reshape(-1, 1)
        state_ref = mpc.calc_ref_trajectory(track_ref, idx, N_mpc).reshape(-1,1) - np.kron(np.ones((N_mpc, 1)), track_ref[idx].reshape(-1, 1))
        
        u = mpc.solve(state_error0,state_ref,A_hat,B_hat,Q,R,Qf,N_mpc)[0:vehicle.n_u]
        # u[0] = u[0] + 0.3

        time_2 = TM.time()
        print('time: ', time_2 - time_1)

        #debug
        # print('i:', i,'u: ', u[0], u[1])

        vehicle.update(u[0], u[1])

        



    plt.plot(x, y, label='real')
    plt.plot(x_mpc, y_mpc, label='mpc')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_dwa()
    # test_mpc()