import math

def ackley(x):
    return -20*math.exp(-0.2*math.sqrt(0.5*(x[0]**2 + x[1]**2))) - math.exp(0.5*(math.cos(2*math.pi*x[0]) + math.cos(2*math.pi*x[1]))) + math.e + 20

def ackley_gradient(x):
    return [-4*math.exp(-0.2*math.sqrt(0.5*(x[0]**2 + x[1]**2))) * x[0] * math.cos(2*math.pi*x[0]) - 4*math.exp(-0.2*math.sqrt(0.5*(x[0]**2 + x[1]**2))) * x[1] * math.cos(2*math.pi*x[1]), -4*math.exp(-0.2*math.sqrt(0.5*(x[0]**2 + x[1]**2))) * x[0] * math.sin(2*math.pi*x[0]) - 4*math.exp(-0.2*math.sqrt(0.5*(x[0]**2 + x[1]**2))) * x[1] * math.sin(2*math.pi*x[1])]