import math
# Rastrigin function
def rastrigin(x):
    return 20 + x[0]**2 + x[1]**2 - 10*(math.cos(2*math.pi*x[0]) + math.cos(2*math.pi*x[1]))

# Gradient of rastrigin function
def rastrigin_gradient(x):
    return [2*x[0] + 2*math.pi*math.sin(2*math.pi*x[0]), 2*x[1] + 2*math.pi*math.sin(2*math.pi*x[1])]