from main import *
import torch as t
import numpy as np


def experiment(N, sigma, d):
    M = 50
    weight_decay = True
    wedecay_lamda = 0.01
    learning_rate = 0.5
    for i in range(M):
        # print('trials:', i)
        # generating the training data set of size N and noise variance sigma
        x_train, y_train = getData(N, sigma)

        # fitting the data to a polynomial of degree d
        Ein = t.zeros(size=(M,))
        Eout = t.zeros(size=(M,))
        weightsum = t.zeros(size=(d + 1,))

        Ein[i], Eout[i], outputs, weight = fitData(x_train, y_train, d, learning_rate, N, 2000, sigma, weight_decay,
                                                   wedecay_lamda)
        weightsum += weight[0]

    # calculate the average of Ein and Eout
    aveEin = Ein.mean()
    aveEout = Eout.mean()

    # calculate the E bias
    weight_ave = weightsum / M
    x_train, y_train = getData(N, sigma)
    inputs_train = poly_build(x_train, d)
    y_predict = polyFunc(inputs_train, weight_ave, N)
    Ebias = getMSE(y_predict, y_train, weight_decay, weight_ave, learning_rate, wedecay_lamda)
    # print('Ein ', aveEin)
    # print('Eout ', aveEout)
    # print('Ebias', Ebias)
    return aveEin, aveEout, Ebias


# ein, eout, ebias = experiment(50, 0.1, 10)
# print(ein, eout, ebias)

def reportData():
    N = [2, 5, 10, 20, 50, 100, 200]
    degree = [i for i in range(1, 21)]
    sigma = [0.01, 0.1, 1]

    nEin = []
    nEout = []
    nEbias = []
    for n in N:
        print('*************************************')
        print('*************************************')
        print('data size: ', n)
        Ein = []
        Eout = []
        Ebias = []
        for d in degree:
            print('degree iter: ', d)
            sigEin = []
            sigEout = []
            sigEbias = []

            for sig in sigma:
                print('sigma iter: ', sig)
                ein, eout, ebias = experiment(n, sig, d)
                Ein.append(ein.item())
                Eout.append(eout.item())
                Ebias.append(ebias.item())

            Ein.append(sigEin)
            Eout.append(sigEout)
            Ebias.append(sigEbias)
        nEin.append(Ein)
        nEout.append(Eout)
        nEbias.append(Ebias)

    In = np.array(nEin)
    Out = np.array(nEout)
    Bias = np.array(nEbias)

    np.save('In.npy', In)
    np.save('out.npy', Out)
    np.save('Bias.npy', Bias)


if __name__ == '__main__':
    reportData()
