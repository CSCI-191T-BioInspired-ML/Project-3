import rastrigin
import random
import math
import matplotlib.pyplot as plt
from GD import gradient_descent
from SA import simulated_annealing
import stats


# Learning Rate tests
lrs = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1] 
LR_ITERATIONS = 10000

# Temperature tests
temps = [0.001, 0.01, 0.1, 1, 10]
TEMP_ITERATIONS = 10000

# Neighboring Solution tests
neighbor_dists = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
NEIGHBOR_ITERATIONS = 10000

# k tests
ks = [1, 3, 5, 9, 15]
K_ITERATIONS = 10000

def main():
    # Gradient Descent (testing learning rate)
    avg_counts = []
    for lr in lrs:
        counts = {}
        for i in range(LR_ITERATIONS):
            x_init = [random.uniform(-5, 5) for _ in range(2)]
            x, x_history = gradient_descent(rastrigin.rastrigin_gradient, x_init, lr, 200)
            counts = stats.proximity_percentage(rastrigin.rastrigin([0,0]), rastrigin.rastrigin(x), counts)
        avg_counts.append(stats.average_proximity_percentage(counts, LR_ITERATIONS))
        
    # Print results
    print("Learning Rate:")
    for i in range(len(lrs)):
        print("{}: {}".format(lrs[i], avg_counts[i]))
    print()

    # Simulated Annealing (testing temperature)
    avg_counts = []
    for temp in temps:
        counts = {}
        for i in range(TEMP_ITERATIONS):
            x_init = [random.uniform(-5, 5) for _ in range(2)]
            x, x_history = simulated_annealing(rastrigin.rastrigin, x_init, temp, 5, 0.01, 200)
            counts = stats.proximity_percentage(rastrigin.rastrigin([0,0]), rastrigin.rastrigin(x), counts)
        avg_counts.append(stats.average_proximity_percentage(counts, TEMP_ITERATIONS))
    
    # Print results
    print("Temperature:")
    for i in range(len(temps)):
        print("{}: {}".format(temps[i], avg_counts[i]))
    print()

    # Simulated Annealing (testing neighboring solution distance)
    avg_counts = []
    for neighbor_dist in neighbor_dists:
        counts = {}
        for i in range(NEIGHBOR_ITERATIONS):
            x_init = [random.uniform(-5, 5) for _ in range(2)]
            x, x_history = simulated_annealing(rastrigin.rastrigin, x_init, 0.01, neighbor_dist, 0.01, 200)
            counts = stats.proximity_percentage(rastrigin.rastrigin([0,0]), rastrigin.rastrigin(x), counts)
        avg_counts.append(stats.average_proximity_percentage(counts, NEIGHBOR_ITERATIONS))

    # Print results
    print("Neighboring Solution Distance:")
    for i in range(len(neighbor_dists)):
        print("{}: {}".format(neighbor_dists[i], avg_counts[i]))
    print()

    # Simulated Annealing (testing k)
    avg_counts = []
    for k in ks:
        counts = {}
        for i in range(K_ITERATIONS):
            x_init = [random.uniform(-5, 5) for _ in range(2)]
            x, x_history = simulated_annealing(rastrigin.rastrigin, x_init, 0.01, k, 0.01, 200)
            counts = stats.proximity_percentage(rastrigin.rastrigin([0,0]), rastrigin.rastrigin(x), counts)
        avg_counts.append(stats.average_proximity_percentage(counts, K_ITERATIONS))

    # Print results
    print("K:")
    for i in range(len(ks)):
        print("{}: {}".format(ks[i], avg_counts[i]))

if __name__ == '__main__':
    main()