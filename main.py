import rastrigin
import sphere
import ackley
import plot
import matplotlib.pyplot as plt
from GD import gradient_descent
from SA import simulated_annealing

import random

def rastrigin_GD_SA():
    # random initial x
    init_x = [random.uniform(-5, 5) for _ in range(2)]
    print("Initial x: ", init_x)

    # Gradient descent
    lr = 0.01
    x, x_history = gradient_descent(rastrigin.rastrigin_gradient, init_x, lr, 200)

    # Plot the results
    plot.plot_results(x_history, 'Rastrigin - GD')
    plot.print_results(x, 'Rastrigin - GD')

    # Plot 3d surface and history
    plot.plot_history(x_history, rastrigin.rastrigin, [-5, 5], [-5, 5], 'Rastrigin - GD')
    plt.show()


    # Simulated annealing
    init_temp = 50
    init_s = init_x
    s, s_history = simulated_annealing(rastrigin.rastrigin, init_s, init_temp, 1, 0.1, 10000)

    # Plot the results
    plot.plot_results(s_history, 'Rastrigin - SA')
    plot.print_results(s, 'Rastrigin - SA') 

    # Plot 3d surface and history
    plot.plot_history(s_history, rastrigin.rastrigin, [-5, 5], [-5, 5], 'Rastrigin - SA')
    plt.show()

def sphere_GD_SA():
    # random initial x
    init_x = [random.uniform(-10, 10) for _ in range(2)]
    print("Initial x: ", init_x)

    # Gradient descent
    lr = 0.01
    x, x_history = gradient_descent(sphere.sphere_gradient, init_x, lr, 200)

    # Plot the results
    plot.plot_results(x_history, 'Sphere - GD')
    plot.print_results(x, 'Sphere - GD')

    # Plot 3d surface and history
    plot.plot_history(x_history, sphere.sphere, [-5, 5], [-5, 5], 'Sphere - GD')
    plt.show()

    # Simulated annealing
    init_temp = 100
    init_s = init_x
    s, s_history = simulated_annealing(sphere.sphere, init_s, init_temp, 1, 0.1, 10000)

    # Plot the results
    plot.plot_results(s_history, 'Sphere - SA')
    plot.print_results(s, 'Sphere - SA') 

    # Plot 3d surface and history
    plot.plot_history(s_history, sphere.sphere, [-5, 5], [-5, 5], 'Sphere - SA')
    plt.show()

def ackley_GD_SA():
    # random initial x
    init_x = [random.uniform(-5, 5) for _ in range(2)]
    print("Initial x: ", init_x)

    # Gradient descent
    lr = 0.01
    x, x_history = gradient_descent(ackley.ackley_gradient, init_x, lr, 200)

    # Plot the results
    plot.plot_results(x_history, 'Ackley - GD')
    plot.print_results(x, 'Ackley - GD')

    # Plot 3d surface and history
    plot.plot_history(x_history, ackley.ackley, [-5, 5], [-5, 5], 'Ackley - GD')
    plt.show()


    # Simulated annealing
    init_temp = 100
    init_s = init_x
    s, s_history = simulated_annealing(ackley.ackley, init_s, init_temp, 1, 0.1, 10000)

    # Plot the results
    plot.plot_results(s_history, 'Ackley - SA')
    plot.print_results(s, 'Ackley - SA') 

    # Plot 3d surface and history
    plot.plot_history(s_history, ackley.ackley, [-5, 5], [-5, 5], 'Ackley - SA')
    plt.show()

if __name__ == '__main__':
    print("Input which function you want to test:")
    print("1. Rastrigin")
    print("2. Sphere")
    print("3. Ackley")

    choice = int(input())
    if choice == 1:
        rastrigin_GD_SA()
    elif choice == 2:
        sphere_GD_SA()
    elif choice == 3:
        ackley_GD_SA()
    else:
        print("Invalid input")
    
    print("Done")