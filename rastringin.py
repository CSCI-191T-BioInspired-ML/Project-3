import math
# Rastringin function
def rastringin(x):
    return 20 + x[0]**2 + x[1]**2 - 10*(math.cos(2*math.pi*x[0]) + math.cos(2*math.pi*x[1]))

# Gradient of Rastringin function
def rastringin_gradient(x):
    return [2*x[0] + 2*math.pi*math.sin(2*math.pi*x[0]), 2*x[1] + 2*math.pi*math.sin(2*math.pi*x[1])]