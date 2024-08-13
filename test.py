import numpy as np
import matplotlib.pyplot as plt
import ctrl

if __name__ == '__main__':
    #generate random path
    v = 0.4
    w = 0.2
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
            car_state.update_vw(0.4, 0.25)

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
    
    # plt.plot(x_noise, y_noise, label='noisy')
    # plt.plot(x_ekf, y_ekf, label='ekf')
    # plt.plot(x_observation, y_observation, label='observation')
    # plt.legend()
    # plt.show()

    #test MPC
    vehicle = ctrl.CarModel()
    Q = np.matrix(np.eye(vehicle.n_x) * 3)
    R = np.matrix(np.eye(vehicle.n_u) * 2)
    Qf = np.matrix(np.eye(vehicle.n_x) * 3)
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
    
    for i in range(n):
        x_mpc[i] = vehicle.x
        y_mpc[i] = vehicle.y

        #debug
        # print('state: ', vehicle.state)
        # print('u: ', vehicle.u)

        A_hat, B_hat = vehicle.stateSpaceModel(vehicle.state, vehicle.u)

        #debug
        # print('A_hat:\n', A_hat, '\n B_hat: \n', B_hat)

        idx = mpc.calc_track_idx(vehicle.x, vehicle.y, track_ref)
        state_error0 = vehicle.state - track_ref[idx].reshape(-1, 1)
        state_ref = mpc.calc_ref_trajectory(track_ref, idx, N_mpc)
        
        u = mpc.solve(state_error0,state_ref,A_hat,B_hat,Q,R,Qf,N_mpc)[0:vehicle.n_u]
        u[0] = u[0] + 0.4

        #debug
        # print('i:', i,'u: ', u[0], u[1])

        vehicle.update(u[0], u[1])

        



    plt.plot(x, y, label='real')
    plt.plot(x_mpc, y_mpc, label='mpc')
    plt.legend()
    plt.show()