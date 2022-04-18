import rastringin
import sphere
import ackley
import plot
from GD import gradient_descent
from SA import simulated_annealing

import random

def main():
    # random initial x
    init_x = [random.uniform(-3, 3) for _ in range(2)]


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

if __name__ == '__main__':
    main()