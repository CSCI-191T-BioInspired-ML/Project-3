import math

def ackley(x):
    return -20*math.exp(-0.2*math.sqrt(0.5*(x[0]**2 + x[1]**2))) - math.exp(0.5*(math.cos(2*math.pi*x[0]) + math.cos(2*math.pi*x[1]))) + math.e + 20

def ackley_gradient(x):
    h = 1e-4
    g0 = (ackley([x[0]+h, x[1]]) - ackley([x[0]-h, x[1]]))/(2*h)
    g1 = (ackley([x[0], x[1]+h]) - ackley([x[0], x[1]-h]))/(2*h)
    return [g0, g1]