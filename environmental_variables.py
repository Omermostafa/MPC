import numpy as np


# Spline functions
def spline_3(x0, x1, y0, y1, dy0, dy1):
    n = 4
    A = np.zeros((n,n))
    b = np.zeros((n,1))
    
    b[0] = y0
    for i in range(n):
        A[0, -1 - i] = x0 ** i

    b[1] = y1
    for i in range(n):
        A[1, -1 - i] = x1 ** i

    b[2] = dy0
    for i in range(n):
        A[2, -1 - i] = i * x0 ** np.clip(i - 1, a_min = 0, a_max = np.inf)

    b[3] = dy1
    for i in range(n):
        A[3, -1 - i] = i * x1 ** (i - 1)

    params = np.linalg.solve(A, b)
    sp_func = lambda x: params.T @ np.array([x ** (n - i - 1) for i in range(n)]).reshape(-1, 1)

    return sp_func

def spline_4(x0, x1, y0, y1, dy0, dy1, x_wp):
    n = 5
    A = np.zeros((n,n))
    b = np.zeros((n,1))

    b[0] = y0
    for i in range(n):
        A[0, -1 - i] = x0 ** i

    b[1] = y1
    for i in range(n):
        A[1, -1 - i] = x1 ** i

    b[2] = dy0
    for i in range(n):
        A[2, -1 - i] = i * x0 ** np.clip(i - 1, a_min = 0, a_max = np.inf)

    b[3] = dy1
    for i in range(n):
        A[3, -1 - i] = i * x1 ** (i - 1)

    b[4] = 0
    for i in range(2, n):
        A[4, -1 - i] = i * (i-1) * x_wp ** (i - 2)
    
    params = np.linalg.solve(A, b)
    sp_func = lambda x: params.T @ np.array([x ** (n - i - 1) for i in range(n)]).reshape(-1, 1)
    return sp_func


# Summer functions
def q_rad_summer(t):
    # Reset everything outside of the 24h period to the 24h period
    rest = t // 24
    t = t - rest * 24

    # Different profiles
    if 0 <= t < 6:
        q_rad = 0.
    
    elif 6 <= t < 12:        
        t0 = 6
        q0 = 0

        t1 = 12
        q1 = 1.3

        dq0 = 0
        dq1 = 0

        t_wp = 10
        sp_func = spline_4(t0, t1, q0, q1, dq0, dq1, t_wp)

        q_rad = sp_func(t)
    
    elif 12 <= t < 18:       
        t0 = 12
        q0 = 1.3

        t1 = 18
        q1 = 1.0

        dq0 = 0
        dq1 = -0.1

        t_wp = 18

        sp_func = spline_4(t0, t1, q0, q1, dq0, dq1, t_wp)

        q_rad = sp_func(t)
    
    elif 18 <= t < 20:
        t0 = 18
        q0 = 1.

        t1 = 20
        q1 = 0

        dq0 = -0.1
        dq1 = 0

        t_wp = 19

        sp_func = spline_4(t0, t1, q0, q1, dq0, dq1, t_wp)

        q_rad = sp_func(t)
    
    elif 20 <= t < 24:
        q_rad = 0.
    
    return float(q_rad)
       
def T_env_summer(t):
    # Reset everything outside of the 24h period to the 24h period
    rest = t // 24
    t = t - rest * 24

    if 0 <= t < 4:
        t0 = 0
        T0 = 20

        t1 = 4
        T1 = 18

        dT0 = 0
        dT1 = -0.5

        sp_func = spline_3(t0, t1, T0, T1, dT0, dT1)
        T_env = sp_func(t)

        T_env = 18.0

    
    elif 4 <= t < 6:
        t0 = 4
        T0 = 18

        t1 = 6
        T1 = 18

        dT0 = 0
        dT1 = 0.8

        t_wp = 2

        sp_func = spline_4(t0, t1, T0, T1, dT0, dT1, t_wp)

        T_env = sp_func(t)

    
    elif 6 <= t < 12:
        t0 = 6
        T0 = 18

        t1 = 12
        T1 = 32

        dT0 = 0.8
        dT1 = 0

        sp_func = spline_3(t0, t1, T0, T1, dT0, dT1)
        T_env = sp_func(t)

    
    elif 12 <= t < 17:
        T_env = 32
    
    elif 17 <= t < 20:       
        t0 = 17
        T0 = 32

        t1 = 20
        T1 = 27

        dT0 = 0
        dT1 = -1.7

        t_wp = 20

        sp_func = spline_4(t0, t1, T0, T1, dT0, dT1, t_wp)

        T_env = sp_func(t)

    
    elif 20 <= t < 24:       
        t0 = 20
        T0 = 27

        t1 = 24
        T1 = 18

        dT0 = -1.7
        dT1 = -0.5

        t_wp = 20

        sp_func = spline_4(t0, t1, T0, T1, dT0, dT1, t_wp)

        T_env = sp_func(t)
    
    T_env += 273.15
    return float(T_env)


# Winter functions
def q_rad_winter(t):
    # Reset everything outside of the 24h period to the 24h period
    rest = t // 24
    t = t - rest * 24

    if 0 <= t < 8:
        q_rad = 0
    
    elif 8 <= t < 10:
        t0 = 8
        q0 = 0

        t1 = 10
        q1 = 0.4

        dq0 = 0
        dq1 = 0.1

        sp_func = spline_3(t0, t1, q0, q1, dq0, dq1)

        q_rad = sp_func(t)
    
    elif 10 <= t < 15:
        t0 = 10
        q0 = 0.4

        t1 = 15
        q1 = 0.4

        dq0 = 0.1
        dq1 = -0.1

        sp_func = spline_3(t0, t1, q0, q1, dq0, dq1)

        q_rad = sp_func(t)
    
    elif 15 <= t < 17:
        t0 = 15
        q0 = 0.4

        t1 = 17
        q1 = 0

        dq0 = -0.1
        dq1 = 0

        sp_func = spline_3(t0, t1, q0, q1, dq0, dq1)

        q_rad = sp_func(t)

    elif 17 <= t < 24:
        q_rad = 0.
    
    return float(q_rad)

def T_env_winter(t):
    # Reset everything outside of the 24h period to the 24h period
    rest = t // 24
    t = t - rest * 24

    if 0 <= t < 6:
        T_env = -10.0

    elif 6 <= t < 10:
        t0 = 6
        T0 = -10

        t1 = 10
        T1 = 4

        dT0 = 0
        dT1 = 1

        sp_func = spline_3(t0, t1, T0, T1, dT0, dT1)
        T_env = sp_func(t)


    elif 10 <= t < 17:
        t0 = 10
        T0 = 4

        t1 = 17
        T1 = 4

        dT0 = 1
        dT1 = -0.5

        sp_func = spline_3(t0, t1, T0, T1, dT0, dT1)
        T_env = sp_func(t)

    elif 17 <= t < 20:
        t0 = 17
        T0 = 4

        t1 = 20
        T1 = -8.5

        dT0 = -0.5
        dT1 = -0.75

        sp_func = spline_3(t0, t1, T0, T1, dT0, dT1)
        T_env = sp_func(t)


    elif 20 <= t < 24:
        t0 = 20
        T0 = -8.5

        t1 = 24
        T1 = -10

        dT0 = -0.75
        dT1 = 0

        sp_func = spline_3(t0, t1, T0, T1, dT0, dT1)
        T_env = sp_func(t)

    T_env += 273.15
    return float(T_env)
