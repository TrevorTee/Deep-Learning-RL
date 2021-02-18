import numpy as np
import torch
import random
import matplotlib.pyplot as plt


def getData(n, sigma, mu=0):
    # generate x by numpy uniform function with (0,1) and N
    x = np.random.uniform(low=0, high=1, size=n)
    theta = 2 * np.pi * x
    y = np.cos(theta) + sigma * np.random.randn(len(x)) + mu
    return torch.from_numpy(x), torch.from_numpy(y)


def getMSE(y_predict, y, weight_decay, w,learning_rate, wedecay_lamda):
    error = (y_predict - y) ** 2
    # calcul the LOSS with weight decay
    if weight_decay:
        w = w **2
        return error.mean() + w.mean()*wedecay_lamda
    return error.mean()


def poly_build(x, d):
    x = x.unsqueeze(1)
    return torch.cat(tensors=[x ** i for i in range(0, d + 1)], dim=1)


def polyFunc(inputs, w, n):
    outputs = inputs * w
    output = [outputs[i].sum().item() for i in range(n)]
    return torch.tensor(output)


def gradWeights(outputs, y_train, inputs, d, n, flag):
    # 将矩阵的各个阶数算清楚，然后重新计算weight
    # ws = (error[i]*inputs[i][:] for i in range(len(error)))
    # GD
    if flag == 1:
        error = y_train - outputs
        grad = torch.zeros(size=(1, d + 1))
        for i in range(n):
            grad += error[i] * inputs[i]
        return grad * 2 / n
    elif flag == 2:         # SGD
        r = np.random.randint(0, n, size=1)
        rand_index = r[0]
        error = y_train[rand_index] - outputs[rand_index]
        grad = error * inputs[rand_index]
        return grad * 2
    elif flag == 3:       # mini-batched SGD
        batch_size = 32
        batch_index = random.sample(population=[i for i in range(n)], k=batch_size)
        error = y_train - outputs
        grad = torch.zeros(size=(1, d + 1))
        for index in batch_index:
            grad += error[index] * inputs[index]
        return grad * 2 / batch_size
    else:
        return


def fitData(x_train, y_train, d, learning_rate, n, num_epochs, sigma, weight_decay, wedecay_lamda):
    # building the poly matrix
    inputs = poly_build(x_train, d)

    # generate the random weights
    # print('degree: ', d)
    w = torch.randn(size=(d+1,))
    flag = 1   # 1: GD; 2: SGD; 3: Mini-batch
    # print(inputs.size())
    # print(w.size())

    # print('Flag', flag)
    for epoch in range(num_epochs):
        # calculate the predicted outputs
        outputs = polyFunc(inputs, w, n)
        if epoch == num_epochs - 1:
            break
        # calculate the gradient descent undated weights
        updated_weights = gradWeights(outputs, y_train, inputs, d, n, flag)
        if weight_decay:
            w = w * (1 - (learning_rate*wedecay_lamda)/(d+1))
        w = w + updated_weights * learning_rate
        # print(w)

    Ein = getMSE(outputs, y_train, weight_decay, w, learning_rate, wedecay_lamda)

    # calculate the Eout by generating the test data set
    n_test = 1000
    x_test, y_test = getData(n_test, sigma)
    inputs_test = poly_build(x_test, d)
    outputs_predict = polyFunc(inputs_test, w=w, n=n_test)
    Eout = getMSE(outputs_predict, y_test, weight_decay, w, learning_rate, wedecay_lamda)

    # print(Ein, Eout)
    return Ein, Eout, outputs, w


def test():
    n = 2000
    sigma = 0.1
    weight_decay = True
    wedecay_lamda = 0.01
    x_train, y_train = getData(n, sigma)
    # 0.1 SGD  0.5 GD
    ein, eout, outputs, weight = fitData(x_train, y_train, 10, 0.5, n, 2000, sigma, weight_decay, wedecay_lamda)
    plt.plot(x_train, y_train, '.')
    plt.plot(x_train, outputs, '.')
    plt.show()


# main function to start
if __name__ == '__main__':
    test()
