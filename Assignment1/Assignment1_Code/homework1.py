import torch
import tensorflow as tf
import numpy as np
import random as rd
from itertools import count
from torch.autograd import Variable
from torch import nn



POLY_DEGREE = 5;
THETA= torch.FloatTensor([3,6,2,1,1,1]).unsqueeze(1)

print(THETA)


def getData(data_size,sigma):
    x = torch.empty(data_size, ).uniform_(0, 1).type(torch.FloatTensor)
    f = torch.cos(2 * np.pi * x)
    Z = torch.normal(0, sigma ** 2, size=(data_size,))
    y = f + Z
    return Variable(x), Variable(y)
test_x = getData(5,1)
print(tf.transpose(test_x[1]))


def make_features(training_data):
    x = training_data[0]
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(0, POLY_DEGREE+1)], 1)


x_featured = make_features(test_x)

print(x_featured)


def hypothesis(X, theta):
    return tf.matmul(X, theta)

y_test = hypothesis(x_featured,THETA)
print ("y-test", y_test)


def getMSE(X,theta,data_size):
    prediction = hypothesis(X, theta)
    mean_square_err = tf.reduce_sum((prediction - test_x[1]) ** 2) / data_size
    return mean_square_err


# MSE_test = getMSE(x_featured,THETA,5)
# print(MSE_test)


def GradientDescent(X, theta, Y, learning_rate, data_size):

    prediction = hypothesis(x_featured,THETA)
    print('prediction', prediction)
    learned_theta = tf.add(THETA, learning_rate * 2 *
                      tf.matmul(tf.transpose(X), (Y - prediction)) /
                      data_size)
    # if self.regularization:
    #     learning = tf.add(learning, -2 * self.learning_rate * self.reg_lambda * self.Theta)
    # operator = self.Theta.assign(learning)
    print("check matrix", tf.matmul(tf.transpose(X), (Y - prediction)))
    print('check X', tf.transpose(X) )
    print('check Y', Y)
    print("Y- prediction", Y - prediction )
    return learned_theta

gd_test = GradientDescent(x_featured,THETA,tf.transpose(test_x[1]),0.04,5)
print(test_x)



print(gd_test)
print(tf.transpose(gd_test))