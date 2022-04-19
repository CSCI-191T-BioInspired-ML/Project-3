import matplotlib.pyplot as plt
import numpy as np

def plot_results(x_history, title):
    plt.plot(x_history)
    plt.xlabel('Step')
    plt.ylabel('x')
    for i in range(len(x_history[0])):
        plt.text(len(x_history), x_history[len(x_history) - 1][i], 'x[' + str(i) + '] = %.2f' % x_history[len(x_history) - 1][i])
    plt.legend(['x0', 'x1'])
    plt.title(title)
    #plt.show()

def print_results(x, title):
    print(title)
    for i, xi in enumerate(x):
        print('x' + str(i) + ' = %.3f' % xi)

def plot_history(x_history, f, xrange, yrange, title):
    xrange = np.arange(xrange[0], xrange[1], 0.2)
    yrange = np.arange(yrange[0], yrange[1], 0.2)
    X, Y = np.meshgrid(xrange, yrange)
    Z = [ np.array(f([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y)) ]
    Z = np.reshape(Z, X.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', alpha=0.1, edgecolor='none', linewidth=0)

    colors = []
    for i in range(len(x_history)):
        colors.append(1 - i / len(x_history))
    ax.scatter([ x[0] for x in x_history ], [ x[1] for x in x_history ], [ f(x) for x in x_history ], c=colors, s=5)

    plt.title(title)
    #plt.show()