import numpy as np
import matplotlib.pyplot as plt

ein = np.load('In_no.npy', allow_pickle=True)
eout = np.load('out_no.npy', allow_pickle=True)
ebias = np.load('Bias_no.npy', allow_pickle= True)

einW = np.load('In.npy', allow_pickle=True)
eoutW = np.load('out.npy', allow_pickle=True)
ebiasW = np.load('Bias.npy', allow_pickle= True)


def getNum(error, Flag):

    re = []
    if Flag == 1:
        n = 6
        index = 0
        e = error[6]
        while index <= 76:
            re.append(e[index])
            index += 4
    elif Flag == 2:
        # get the data by Sample size
        index = 0 + 4*6 # select degree 6
        for i in range(7):
            re.append(error[i][index])
    elif Flag == 3:
        for i in [24, 25, 26]:
            re.append(error[6][i])

    result = []
    for i in range(len(re)):
        if np.isinf(re[i]) or np.isnan(re[i]):
            continue
        result.append(re[i])
    return result




# Ein = getNum(ein, 1)
# EinW = getNum(einW, 1)
# Eout = getNum(eout, 1)
# EoutW = getNum(eoutW, 1)
# Ebias = getNum(ebias, 1)
# EbiasW = getNum(ebiasW, 1)
# fig = plt.figure()
# fig.suptitle("Comparing By Weight Decay")
# plt.subplot(2, 1, 1)
# plt.ylabel("Ebias MSE")
# plt.plot(Ebias,color='red', label="Ebias Without Weight Decay")
# plt.legend()
# plt.plot(EbiasW, color='red', linestyle='--', label="Ebias With Weight Decay")
# plt.legend()
# plt.subplot(2, 1, 2)
# plt.ylabel("Ein and Eout MSE")
# plt.xlabel("Sample Size=200, Sigma=0.01")
# plt.plot(Ein,color='orange', label="Ein Without Weight Decay")
# plt.legend()
# plt.plot(EinW, color='orange', linestyle='--', label="Ein With Weight Decay")
# plt.legend()
# plt.plot(Eout,color='blue', label="Eout Without Weight Decay")
# plt.legend()
# plt.plot(EoutW, color='blue', linestyle='--', label="Eout With Weight Decay")
# plt.legend()
# plt.show()


Ein = getNum(ein, 3)
EinW = getNum(einW, 3)
Eout = getNum(eout, 3)
EoutW = getNum(eoutW, 3)
Ebias = getNum(ebias, 3)
EbiasW = getNum(ebiasW, 3)
fig = plt.figure()
fig.suptitle("Comparing By Weight Decay")
plt.subplot(2, 1, 1)
plt.ylabel("Ebias MSE")
plt.plot(Ebias,color='red', label="Ebias Without Weight Decay")
plt.legend()
plt.plot(EbiasW, color='red', linestyle='--', label="Ebias With Weight Decay")
plt.legend()
plt.subplot(2, 1, 2)
plt.ylabel("Ein and Eout MSE")
plt.xlabel("Degree=6, Sample Size=6")
plt.plot(Ein,color='orange', label="Ein Without Weight Decay")
plt.legend()
plt.plot(EinW, color='orange', linestyle='--', label="Ein With Weight Decay")
plt.legend()
plt.plot(Eout,color='blue', label="Eout Without Weight Decay")
plt.legend()
plt.plot(EoutW, color='blue', linestyle='--', label="Eout With Weight Decay")
plt.legend()
plt.show()