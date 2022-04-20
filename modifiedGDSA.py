from GD import gradient_descent
from SA import modifiedSA, simulated_annealing
import rastrigin
import plot
import matplotlib.pyplot as plt

import random

def modifiedGDSA(GD_settings, SA_settings, function, exploring_range, gd_runs):
    (f, init_x, lr, step_num) = GD_settings
    (init_temp, k, neighbor_range, step_num) = SA_settings

    # run gd multiple times
    gd_minimums = []
    for i in range(gd_runs):
        rand_x = [random.uniform(exploring_range[0], exploring_range[1]) for _ in range(len(init_x))]
        x, x_history = gradient_descent(f, rand_x, lr, step_num)
        gd_minimums.append((x, function(x)))

    # store the best x and its value for the function in a list
    # remove minimums that have the same value
    gd_minimums.sort(key=lambda x: x[1])
    gd_minimums = [gd_minimums[0]] + [x for x in gd_minimums[1:] if x[1] != gd_minimums[0][1]]

    # run modified SA where the xs in the list have higher temperatures
    s, s_history = modifiedSA(function, init_x, init_temp, k, neighbor_range, step_num, gd_minimums)
    #s, s_history = simulated_annealing(function, init_x, init_temp, k, neighbor_range, step_num)

    return (s, s_history, gd_minimums)

if __name__ == "__main__":
    # GD settings
    f = rastrigin.rastrigin_gradient
    function = rastrigin.rastrigin
    lr = 0.01
    step_num = 200
    init_x = [random.uniform(-5, 5) for _ in range(2)]
    GD_settings = (f, init_x, lr, step_num)

    # SA settings
    init_temp = 100
    k = 1
    neighbor_range = 0.1
    step_num = 10000
    SA_settings = (init_temp, k, neighbor_range, step_num)

    exploring_range = [-5, 5]
    gd_runs = 100
    s, s_history, minimums = modifiedGDSA(GD_settings, SA_settings, function, exploring_range, gd_runs)

    # Plot the results
    plot.plot_results(s_history, 'Rastrigin - GDSA')
    plot.plot_history(s_history, function, exploring_range, exploring_range, 'Rastrigin - GDSA')
    plot.print_results(s, 'Rastrigin - GDSA')
    plt.show()