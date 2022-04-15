import rastringin as rastringin
import matplotlib.pyplot as plt
import random
import numpy as np
import math

# GD algorithm
def gradient_descent(gradient_func, init_x, lr, step_num=150):
    x = init_x
    x_history = [x]
    for i in range(step_num):
        grad = gradient_func(x)
        x = [x[j] - lr * grad[j] for j in range(len(x))]
        x_history.append(x)

    return x, x_history

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

def main():
    # random initial x
    init_x = [random.uniform(-5, 5) for _ in range(2)]
    lr = 0.01
    x, x_history = gradient_descent(rastringin.gradient, init_x, lr, 200)

    # Plot the results
    plt.plot(x_history)
    plt.xlabel('Step')
    plt.ylabel('x')
    plt.text(len(x_history)/2, x[0], 'x0 = %.2f' % x[0])
    plt.text(len(x_history), x[1], 'x1 = %.2f' % x[1])
    plt.show()

    # plot points on 3d surface of rastringin function
    x_range = np.arange(-5, 5, 0.1)
    y_range = np.arange(-5, 5, 0.1)
    X, Y = np.meshgrid(x_range, y_range)
    Z = [ np.array(rastringin.rastringin([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y)) ]
    Z = np.reshape(Z, X.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    plt.show()


    # Simulated annealing
    init_temp = 100
    init_s = init_x
    s, s_history = simulated_annealing(rastringin.rastringin, init_s, init_temp, 1, 0.1, 10000)

    # Plot the results
    # plt.plot(s_history)
    # plt.xlabel('Step')
    # plt.ylabel('s')
    # plt.text(len(s_history)/2, s[0], 'x0 = %.2f' % s[0])
    # plt.text(len(s_history), s[1], 'x1 = %.2f' % s[1])
    # plt.show()

    # 

    # # Plot points on the 3d surface
    # x_range = np.arange(-5, 5, 0.1)
    # y_range = np.arange(-5, 5, 0.1)
    # x_grid, y_grid = np.meshgrid(x_range, y_range)
    # z_grid = [[rastringin.rastringin([x_grid[i][j], y_grid[i][j]]) for j in range(len(y_range))] for i in range(len(x_range))]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(x_grid, y_grid, z_grid, cmap='rainbow')
    # ax.scatter(s[0], s[1], rastringin.rastringin(s), c='r', marker='o')
    # plt.show()

if __name__ == '__main__':
    main()