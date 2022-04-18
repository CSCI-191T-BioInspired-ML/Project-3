import rastringin
import sphere
import ackley
import plot
from GD import gradient_descent
from SA import simulated_annealing

import random

def rastringin_GD_SA():
    # random initial x
    init_x = [random.uniform(-3, 3) for _ in range(2)]
    print("Initial x: ", init_x)

    # Gradient descent
    lr = 0.01
    x, x_history = gradient_descent(rastringin.rastringin_gradient, init_x, lr, 200)

    # Plot the results
    plot.plot_results(x_history, 'Rastringin - GD')
    plot.print_results(x, 'Rastringin - GD')

    # Plot 3d surface and history
    plot.plot_history(x_history, rastringin.rastringin, [-5, 5], [-5, 5], 'Rastringin - GD')


    # Simulated annealing
    init_temp = 100
    init_s = init_x
    s, s_history = simulated_annealing(rastringin.rastringin, init_s, init_temp, 1, 0.01, 10000)

    # Plot the results
    plot.plot_results(s_history, 'Rastringin - SA')
    plot.print_results(s, 'Rastringin - SA') 

    # Plot 3d surface and history
    plot.plot_history(s_history, rastringin.rastringin, [-5, 5], [-5, 5], 'Rastringin - SA')

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


    # Simulated annealing
    init_temp = 100
    init_s = init_x
    s, s_history = simulated_annealing(sphere.sphere, init_s, init_temp, 1, 0.1, 10000)

    # Plot the results
    plot.plot_results(s_history, 'Sphere - SA')
    plot.print_results(s, 'Sphere - SA') 

    # Plot 3d surface and history
    plot.plot_history(s_history, sphere.sphere, [-5, 5], [-5, 5], 'Sphere - SA')

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


    # Simulated annealing
    init_temp = 100
    init_s = init_x
    s, s_history = simulated_annealing(ackley.ackley, init_s, init_temp, 1, 0.05, 10000)

    # Plot the results
    plot.plot_results(s_history, 'Ackley - SA')
    plot.print_results(s, 'Ackley - SA') 

    # Plot 3d surface and history
    plot.plot_history(s_history, ackley.ackley, [-5, 5], [-5, 5], 'Ackley - SA')

if __name__ == '__main__':
    print("Input which function you want to test:")
    print("1. Rastringin")
    print("2. Sphere")
    print("3. Ackley")

    choice = int(input())
    if choice == 1:
        rastringin_GD_SA()
    elif choice == 2:
        sphere_GD_SA()
    elif choice == 3:
        ackley_GD_SA()
    else:
        print("Invalid input")
    
    print("Done")