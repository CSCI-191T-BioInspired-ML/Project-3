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

# Modifed Simulated Annealing where the xs in the list have higher temperatures
# ignore the first min because it could be the global minimum
def modifiedSA(func, init_x, init_temp, k, neighbor_range=0.1, step_num=150, gd_mins = []):
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
            if (len(gd_mins) > 1):
                # get rid of the first min (potential global minimum)
                gd_mins.pop(0)
                # if the new solution is very close to one of the gd minimums, increase the probability of acceptance
                for gd_min in gd_mins:
                    if (abs(s_new[0] - gd_min[0][0]) < 0.1 and abs(s_new[1] - gd_min[0][1]) < 0.1):
                        p = math.exp((E_s - E_s_new) / (T * 10))
                        break
            if random.random() < p:
                s = s_new
        
        x_history.append(s)
    
    return s, x_history