import math
import random

# Logarithmic multiplicative cooling schedule
def cooling_schedule(init_T, curr_step, k):
    return init_T / (1 + k * math.log(curr_step + 1))

# Neighbor solution function
def neighbor(x, neighbor_range=1):
    return [x[i] + random.uniform(-neighbor_range, neighbor_range) for i in range(len(x))]

# Energy function (minimization)
def energy(f, x):
    return f(x)

# Simulated annealing algorithm
def simulated_annealing(func, init_x, init_temp, k, neighbor_range=0.1, step_num=150):
    s = init_x
    T = init_temp
    x_history = [s]
    for i in range(step_num):
        T = cooling_schedule(init_temp, i, k)
        s_new = neighbor(s, neighbor_range)

        E_s = energy(func, s)
        E_s_new = energy(func, s_new)

        if E_s >= E_s_new:
            s = s_new
        else:
            p = math.exp((E_s - E_s_new) / T)
            if random.random() < p:
                s = s_new
        
        x_history.append(s)
    
    return s, x_history