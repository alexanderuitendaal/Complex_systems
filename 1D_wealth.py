import numpy as np
import matplotlib.pyplot as plt
import random

N = 1000
T = 100000
w = 50 + 50*np.random.rand(N)

neighbours = [-1, 1]
fracs = np.random.rand(T)

wplot = np.sort(w)
x = np.arange(N)
plt.figure(0)
plt.plot(x, wplot, color='r', label='T=0')

def rank_size_plot(data, label=None, c=1.0):
    fig, ax = plt.subplots()
    w = - np.sort(- data)                  # Reverse sort
    w = w[:int(len(w) * c)]                # extract top (c * 100)%
    rank_data = np.arange(len(w)) + 1
    size_data = w
    ax.loglog(rank_data, size_data, 'o', markersize=3.0, alpha=0.5, label=label)
    if label:
        ax.legend()
    ax.set_xlabel("log rank")
    ax.set_ylabel("log size")
    plt.show()

def lorenz_curve(X):
    X_lorenz = X.cumsum() / X.sum()
    X_lorenz = np.insert(X_lorenz, 0, 0)
    fig, ax = plt.subplots(figsize=[6,6])
    ax.scatter(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz,
               marker='x', color='darkgreen', s=100)
    ## line plot of equality
    ax.plot([0,1], [0,1], color='k')
    plt.show()


for iter in range(T):
    pos = np.argmin(w)
    pos2 = pos + random.choice(neighbours)
    if pos2 == N:
        pos2 = 0
    if pos2 == -1:
        pos2=N-1
    sum = w[pos]+w[pos2]
    e = fracs[iter]
    w[pos] = e*sum
    w[pos2] = (1-e)*sum

wplot = np.sort(w)
plt.plot(x, wplot, color='b', label=f'T={T}')
plt.xlabel('Agents')
plt.ylabel('Wealth')
plt.legend()
plt.show()

lorenz_curve(wplot)
rank_size_plot(wplot)
