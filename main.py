import rastringin
import matplotlib.pyplot as plt
import random
import numpy as np

def rastringin_gradient_descent(f, init_x, lr, step_num=150):
    x = init_x
    x_history = [x]
    for i in range(step_num):
        grad = rastringin.rastringin_gradient(x)
        x = [x[j] - lr * grad[j] for j in range(len(x))]
        x_history.append(x)

    return x, x_history

def main():
    # random initial x
    init_x = [random.uniform(-5, 5) for _ in range(2)]
    lr = 0.01
    x, x_history = rastringin_gradient_descent(rastringin.rastringin, init_x, lr, step_num=200)

    # Plot the results
    plt.plot(x_history)
    plt.xlabel('Step')
    plt.ylabel('x')
    plt.text(90, x[1], 'x = %.2f, %.2f' % (x[0], x[1]))
    plt.show()

if __name__ == '__main__':
    main()