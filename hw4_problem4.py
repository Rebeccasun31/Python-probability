import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

myfile = open('hw4_problem4_data.pickle', 'rb')
mydict = pickle.load(myfile)

X_train = mydict['X_train']
X_test = mydict['X_test']
Y_train = mydict['Y_train']
Y_test = mydict['Y_test']

predictive_mean = np.empty(X_test.shape[0])
predictive_std = np.empty(X_test.shape[0])

sigma = 0.1
sigma_f = 1.0
ls = 0.5

N = X_train.shape[0]
I = np.identity(N)
Y = np.empty((N, 1))
for i in range(N):
    Y[i][0] = Y_train[i]

for k in range(X_test.shape[0]):
    cov = np.empty((N + 1, N + 1))
    X = np.empty(N + 1)
    for i in range(N):
        X[i] = X_train[i]
    X[N] = X_test[k]

    for i in range(N + 1):
        for j in range(N + 1):
            if i == j:
                delta = 1
            else:
                delta = 0
            cov[i][j] = (sigma_f ** 2) * np.exp(-((X[i] - X[j]) ** 2) / (2 * (ls ** 2))) + (sigma ** 2) * delta

    tmp1 = np.empty((1, N))
    for i in range(N):
        tmp1[0][i] = cov[N][i]
    tmp2 = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            tmp2[i][j] = cov[i][j]
    tmp3 = linalg.inv(tmp2 + ((sigma ** 2) * I))
    tmp4 = np.matmul(tmp1, tmp3)

    predictive_mean[k] = np.matmul(tmp4, Y)
    predictive_std[k] = np.sqrt(cov[N][N] - np.matmul(tmp4, np.transpose(tmp1)))

# Visualize the training data, testing data, and predictive distributions
fig = plt.figure()
plt.plot(X_train, Y_train, linestyle='', color='b',
         markersize=5, marker='+', label="Training data")
plt.plot(X_test, Y_test, linestyle='', color='orange',
         markersize=2, marker='^', label="Testing data")
plt.plot(X_test, predictive_mean, linestyle=':', color='green')
plt.fill_between(X_test.flatten(), predictive_mean - predictive_std,
                 predictive_mean + predictive_std, color='green', alpha=0.13)
plt.fill_between(X_test.flatten(), predictive_mean - 2*predictive_std,
                 predictive_mean + 2*predictive_std, color='green', alpha=0.07)
plt.fill_between(X_test.flatten(), predictive_mean - 3*predictive_std,
                 predictive_mean + 3*predictive_std, color='green', alpha=0.04)
plt.xlabel("X")
plt.ylabel("Y")

plt.show()