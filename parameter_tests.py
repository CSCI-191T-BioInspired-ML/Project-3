import rastringin
import random
import math
import matplotlib.pyplot as plt
from GD import gradient_descent
from SA import simulated_annealing


# Learning Rate tests
lrs = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1] 
LR_ITERATIONS = 10000

# Temperature tests
temps = [1, 10, 100, 500, 1000, 1500, 2000, 2500, 3000]
TEMP_ITERATIONS = 10000

def main():
    # Gradient Descent (testing learning rate)
    lr_avg_distances = []
    for lr in lrs:
        sum_distances = (0, 0)
        for i in range(LR_ITERATIONS):
            x_init = [random.uniform(-3, 3) for _ in range(2)]
            x, x_history = gradient_descent(rastringin.rastringin_gradient, x_init, lr, 200)
            sum_distances = (sum_distances[0] + x[0], sum_distances[1] + x[1])
        avg_distances = (sum_distances[0] / LR_ITERATIONS, sum_distances[1] / LR_ITERATIONS)

        # Euclidean distance from (0, 0)
        euclid_distance = math.sqrt(avg_distances[0] ** 2 + avg_distances[1] ** 2)

        lr_avg_distances.append(euclid_distance)
    
    # Print results
    print("Learning Rate:")
    for i in range(len(lrs)):
        print("{}: {}".format(lrs[i], lr_avg_distances[i]))

    # Plot the results
    # Use log scale for learning rate
    plt.plot(lrs, lr_avg_distances)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Euclidean Distance')
    plt.title('Rastringin - GD')
    plt.show()

    
    # Simulated Annealing (testing temperature)
    temp_avg_distances = []
    for temp in temps:
        sum_distances = (0, 0)
        for i in range(TEMP_ITERATIONS):
            x_init = [random.uniform(-3, 3) for _ in range(2)]
            x, x_history = simulated_annealing(rastringin.rastringin, x_init, temp, 1, 0.01, 100)
            sum_distances = (sum_distances[0] + x[0], sum_distances[1] + x[1])
        avg_distances = (sum_distances[0] / TEMP_ITERATIONS, sum_distances[1] / TEMP_ITERATIONS)

        # Euclidean distance from (0, 0)
        euclid_distance = math.sqrt(avg_distances[0] ** 2 + avg_distances[1] ** 2)

        temp_avg_distances.append(euclid_distance)

    # Print results
    print("Temperature:")
    for i in range(len(temps)):
        print("{}: {}".format(temps[i], temp_avg_distances[i]))

    # Plot the results
    # Use log scale for temperature
    plt.plot(temps, temp_avg_distances)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Temperature')
    plt.ylabel('Euclidean Distance')
    plt.title('Rastringin - SA')
    plt.show()

if __name__ == '__main__':
    main()