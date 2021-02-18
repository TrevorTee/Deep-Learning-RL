import numpy as np
import matplotlib.pyplot as plt


def plotMC():
    einc = ein[6]
    eoutc = eout[6]
    ebiasc = ebias[6]

    index = 0
    Ein = []
    Eout = []
    Ebias = []
    while index <= 76:
        Ein.append(einc[index])
        Eout.append(eoutc[index])
        Ebias.append(ebiasc[index])
        index += 4
    Eins = []
    Eouts = []
    Ebiass = []
    for i in range(20):
        if np.isnan(Ein[i]) or np.isinf(Ein[i]):
            continue
        Eins.append(Ein[i])
        Eouts.append(Eout[i])
        Ebiass.append(Ebias[i])


    print(Eins)
    x = np.linspace(1, 20, len(Eins))
    print(x)
    plt.plot(x, Eins, label="Ein")
    plt.legend()
    plt.plot(x, Eouts, label="Eout")
    plt.legend()
    plt.plot(x, Ebiass, label="Ebias")
    plt.legend()


def plotSampleSize():
    x = [2, 5, 10, 20, 50, 100, 200]
    index = 0+4*6
    Ein = []
    Eout = []
    Ebias = []
    for i in range(7):
        Ein.append(ein[i][index])
        Eout.append(eout[i][index])
        Ebias.append(ebias[i][index])

    plt.plot(x, Ein, label="Ein")
    plt.legend()
    plt.plot(x, Eout, label="Eout")
    plt.legend()
    plt.plot(x, Ebias, label="Ebias")
    plt.legend()
    # ax = plt.gca()
    # ax.yaxis.set_major_locator(MultipleLocator(0.001))
    # plt.ylim(0, 0.02)
    # plt.show()


def plotSigma():
    index = [24, 25, 26]
    einc = ein[6]
    eoutc = eout[6]
    ebiasc = ebias[6]
    Ein = []
    Eout = []
    Ebias = []
    for ind in index:
        Ein.append(einc[ind])
        Eout.append(einc[ind])
        Ebias.append(ebiasc[ind])

    x = np.linspace(0, 1, 3)
    plt.plot(x, Ein, label="Ein")
    plt.legend()
    plt.plot(x, Eout, label="Eout")
    plt.legend()
    plt.plot(x, Ebias, label="Ebias")
    plt.legend()




ein = np.load('In_no.npy', allow_pickle=True)
eout = np.load('out_no.npy', allow_pickle=True)
ebias = np.load('Bias_no.npy', allow_pickle= True)

fig = plt.figure(6)
fig.suptitle("With/Without Weight Decay")
plt.subplot(2, 3, 1)
plt.ylabel("MSE")
plt.xlabel("Sample Size (N)")
plotSampleSize()
plt.subplot(2, 3, 2)
plt.xlabel("Polynomials Degree (d) ")
plotMC()
plt.subplot(2, 3, 3)
plt.xlabel("Noise (sigma)")
plotSigma()


ein = np.load('In.npy', allow_pickle=True)
eout = np.load('out.npy', allow_pickle=True)
ebias = np.load('Bias.npy', allow_pickle= True)


plt.subplot(2, 3, 4)
plt.ylabel("MSE")
plt.xlabel("Sample Size (N)")
plotSampleSize()
plt.subplot(2, 3, 5)
plt.xlabel("Polynomials Degree (d) ")
plotMC()
plt.subplot(2, 3, 6)
plt.xlabel("Noise (sigma)")
plotSigma()
plt.show()




