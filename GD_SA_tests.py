import math
import random

import rastrigin
import sphere
import ackley
from GD import gradient_descent
from SA import simulated_annealing

import matplotlib.pyplot as plt
import stats

ITERATIONS = 10000
ranges = [10, 5, 3, 2, 1]

def rastrigin_tests():
    # GD
    counts = {}
    for i in range(ITERATIONS):
        init_x = [random.uniform(-5, 5) for _ in range(2)]
        x, x_history = gradient_descent(rastrigin.rastrigin_gradient, init_x, 0.01, 200)
        counts = stats.proximity_percentage(rastrigin.rastrigin([0,0]), rastrigin.rastrigin(x), counts)
    for i in ranges:
        print("Rastrigin - GD: <", i, ": ", counts.get(i, 0) / ITERATIONS * 100, "%")
    print()

    # SA
    counts = {}
    for i in range(ITERATIONS):
        init_temp = 0.01
        init_s = [random.uniform(-5, 5) for _ in range(2)]
        s, s_history = simulated_annealing(rastrigin.rastrigin, init_s, init_temp, 5, 0.1, 200)
        counts = stats.proximity_percentage(rastrigin.rastrigin([0,0]), rastrigin.rastrigin(s), counts)
    for i in ranges:
        print("Rastrigin - SA: <", i, ": ", counts.get(i, 0) / ITERATIONS * 100, "%")
    print()

def sphere_tests():
    # GD
    counts = {}
    for i in range(ITERATIONS):
        init_x = [random.uniform(-10, 10) for _ in range(2)]
        x, x_history = gradient_descent(sphere.sphere_gradient, init_x, 0.01, 200)
        counts = stats.proximity_percentage(sphere.sphere([0,0]), sphere.sphere(x), counts)
    for i in ranges:
        print("Sphere - GD: <", i, ": ", counts.get(i, 0) / ITERATIONS * 100, "%")
    print()

    # SA
    counts = {}
    for i in range(ITERATIONS):
        init_temp = 0.01
        init_s = [random.uniform(-10, 10) for _ in range(2)]
        s, s_history = simulated_annealing(sphere.sphere, init_s, init_temp, 5, 0.1, 200)
        counts = stats.proximity_percentage(sphere.sphere([0,0]), sphere.sphere(s), counts)
    for i in ranges:
        print("Sphere - SA: <", i, ": ", counts.get(i, 0) / ITERATIONS * 100, "%")
    print()

def ackley_tests():
    # GD
    counts = {}
    for i in range(ITERATIONS):
        init_x = [random.uniform(-5, 5) for _ in range(2)]
        x, x_history = gradient_descent(ackley.ackley_gradient, init_x, 0.01, 200)
        counts = stats.proximity_percentage(ackley.ackley([0,0]), ackley.ackley(x), counts)
    for i in ranges:
        print("Ackley - GD: <", i, ": ", counts.get(i, 0) / ITERATIONS * 100, "%")
    print()

    # SA
    counts = {}
    for i in range(ITERATIONS):
        init_temp = 0.01
        init_s = [random.uniform(-5, 5) for _ in range(2)]
        s, s_history = simulated_annealing(ackley.ackley, init_s, init_temp, 5, 0.1, 200)
        counts = stats.proximity_percentage(ackley.ackley([0,0]), ackley.ackley(s), counts)
    for i in ranges:
        print("Ackley - SA: <", i, ": ", counts.get(i, 0) / ITERATIONS * 100, "%")
    print()

if __name__ == "__main__":
    print("Input the function you want to test:")
    print("1. Rastrigin")
    print("2. Sphere")
    print("3. Ackley")
    choice = int(input())
    if choice == 1:
        rastrigin_tests()
    elif choice == 2:
        sphere_tests()
    elif choice == 3:
        ackley_tests()
    else:
        print("Invalid input")
    
    print("Done")
