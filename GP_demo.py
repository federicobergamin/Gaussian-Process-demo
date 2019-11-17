'''
We try to create an interactive demo of a gaussian process.
At the beginning we will have only the prior and then we can click
on the plot, select a point and refit the gaussian process
'''

import matplotlib.pyplot as plt
import numpy as np
import argparse
import threading

SINGLE_DOUBLE_CLICK_INTERVAL = 0.2
t = None

#TODO: develop argparse
parser = argparse.ArgumentParser(description='GAUSSIAN PROCESS DEMO')



## kernel definition
def squared_exponential_kernel(a, b, lengthscale, variance):
    """ GP squared exponential kernel """
    # compute the pairwise distance between all the point
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return variance * np.exp(-.5 * (1/lengthscale**2) * sqdist)

## we should try to compute the periodic kernel
# def periodic_kernel(a, b, lengthscale, variance, period):
#     if len(a) == 0 and len(b)!=0:
#         return np.array([[]] * len(b)).reshape(-1,len(b))
#     elif len(a) == 0 and len(b) == 0:
#         return np.array([])
#     else:
#         abs_dist = np.zeros((len(a),len(b)))
#         for i in range(len(a)):
#             abs_dist[i,:] = np.abs(a[i] - b).flat
#
#     numerator = 2 * (np.sin((np.pi * abs_dist) / period)**2)
#     return (variance * np.exp(-1 * (numerator/lengthscale**2)))

def fit_GP(X, y, Xtest, kernel, lengthscale, kernel_variance, noise_variance, period=1):
    ## we should standardize the data
    X = np.array(X).reshape(-1,1)
    y = np.array(y)
    K = kernel(X, X, lengthscale, kernel_variance)
    L = np.linalg.cholesky(K + noise_variance * np.eye(len(X)))

    # compute the mean at our test points.
    Lk = np.linalg.solve(L, kernel(X, Xtest, lengthscale, kernel_variance))
    mu = np.dot(Lk.T, np.linalg.solve(L, y))

    # compute the variance at our test points.
    K_ = kernel(Xtest, Xtest, lengthscale, kernel_variance)
    # print(K_.shape)
    # s2 = np.diag(K_) - np.sum(Lk ** 2, axis=0)
    covariance = K_ - Lk.T @ Lk
    # print(covariance.shape)
    # s = np.sqrt(s2)
    return mu, covariance


## parameters definition
lengthscale = 1 # determines the lengths of the wiggle
kernel_variance = 1 # scale factor
noise_var = 0.005
period = 3
n_test_point = 100
n_samples = 15
Xtrain = []
ytrain = []

Xtest = np.linspace(-5, 5, n_test_point).reshape(-1,1)

def onclick(event):
    global t
    if t is None:
        t = threading.Timer(SINGLE_DOUBLE_CLICK_INTERVAL, on_singleclick, [event])
        t.start()
    if event.dblclick:
        t.cancel()
        on_dblclick(event)

def on_singleclick(event):
    global t
    Xtrain.append(event.xdata)
    ytrain.append(event.ydata)
    #clear frame
    plt.clf()
    # we have to refit the GP
    mu, covariance = fit_GP(Xtrain, ytrain, Xtest, squared_exponential_kernel, lengthscale, kernel_variance, noise_var)
    var = np.sqrt(np.diag(covariance))
    plt.plot(Xtrain, ytrain, 'ro')
    plt.gca().fill_between(Xtest.flat, mu - 3 * var, mu + 3 * var,  color='lightblue', alpha=0.5)
    plt.plot(Xtest, mu, 'blue')
    plt.axis([-5, 5, -5, 5])
    plt.draw() #redraw
    t = None

def on_dblclick(event):
    global t
    ## we want the mean + the std deviation but also some samples from the posterior
    # clear frame
    plt.clf()
    # we have to refit the GP
    mu, covariance = fit_GP(Xtrain, ytrain, Xtest, squared_exponential_kernel, lengthscale, kernel_variance, noise_var)
    # we should get the var
    var = np.sqrt(np.diag(covariance))
    # and we have to sample for it
    samples = np.random.multivariate_normal(mu.reshape(-1), covariance, n_samples)  # SxM
    plt.plot(Xtrain, ytrain, 'ro')
    plt.gca().fill_between(Xtest.flat, mu - 2 * var, mu + 2 * var, color='lightblue', alpha=0.5)
    plt.plot(Xtest, mu, 'blue')
    for sample_id in range(n_samples):
        plt.plot(Xtest, samples[sample_id])
    plt.axis([-5, 5, -5, 5])
    plt.draw()  # redraw
    t = None

fig,ax=plt.subplots()
mu, covariance = fit_GP(Xtrain, ytrain, Xtest, squared_exponential_kernel, lengthscale, kernel_variance, noise_var)
var = np.sqrt(np.diag(covariance))
plt.plot(Xtrain, ytrain, 'ro')
plt.gca().fill_between(Xtest.flat, mu - 2 * var, mu + 2 * var,  color='lightblue', alpha=0.5)
plt.plot(Xtest, mu, 'blue')
plt.axis([-5, 5, -5, 5])
fig.canvas.mpl_connect('button_press_event',onclick)
plt.show()
plt.draw()