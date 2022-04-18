import math

# Sphere function
def sphere(x):
    return sum([x[i]**2 for i in range(len(x))])

# Gradient of Sphere function
def sphere_gradient(x):
    return [2*x[i] for i in range(len(x))]