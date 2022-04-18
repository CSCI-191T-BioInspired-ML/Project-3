# GD algorithm
def gradient_descent(gradient_func, init_x, lr, step_num=150):
    x = init_x
    x_history = [x]
    for i in range(step_num):
        grad = gradient_func(x)
        x = [x[j] - lr * grad[j] for j in range(len(x))]
        x_history.append(x)

    return x, x_history