import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from tree.metrics import *
from sklearn.datasets import make_regression, make_classification

np.random.seed(42)
num_average_time = 1  # Number of times to run each experiment to calculate the average values

# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
def calc_time(type_in, type_out) :
    train = []
    test = []
    x = np.zeros(25)

    for i in range(1, 6) :
        for j in range(1, 6) :
            x[5*i+j-6] = 20*i*np.log(10*i)*j
            if type_out == 'classifier' :
                X, y = make_classification(n_samples=10*i, n_features=2*j, n_redundant=0, n_informative=2*j, n_classes=2)
            else :
                X, y = make_regression(n_samples=10*i, n_features=2*j)
            if type_in == 'Real' :
                X = pd.DataFrame(X, columns = [f"f_{k}" for k in range(2*j)])
            else :
                X = pd.DataFrame(X, columns = [f"f_{k}" for k in range(2*j)], dtype='category')
            y = pd.Series(y)
            X_train = X[:7*i]
            X_test = X[7*i:].reset_index(drop=True)
            y_train = y[:7*i]
            y_test = y[7*i:].reset_index(drop=True)
            tree = DecisionTree('entropy')
            st = time.time()
            tree.fit(X_train, y_train)
            end = time.time()
            train.append(end-st)
            st = time.time()
            tree.predict(X_test)
            end = time.time()
            test.append(end-st)

    return x, train, test

# Function to plot the results
def plot_time(x, Tr, Ts, title) :
    plt.figure()
    plt.scatter(x,Tr)
    plt.title(title)
    plt.ylabel("Train time")
    plt.xlabel("N*log(N)*M")
    plt.show()
    plt.figure()
    plt.scatter(x,Ts)
    plt.title(title)
    plt.ylabel("Test time")
    plt.xlabel("N*log(N)*M")
    plt.show()

# Run the functions, Learn the DTs and Show the results/plots
    
# Case 1 : Real input Discrete Output
Tr = np.zeros(25)
Ts = np.zeros(25)
for i in range(num_average_time) :
    x, tr, ts = calc_time('real', 'classifier')
    Tr += tr
    Ts += ts
plot_time(x,Tr,Ts,"Real input Discrete Output")


# Case 2 : Real input Real Output
Tr = np.zeros(25)
Ts = np.zeros(25)
for i in range(num_average_time) :
    x, tr, ts = calc_time('real', 'regressor')
    Tr += tr
    Ts += ts
plot_time(x,Tr,Ts,"Real input Real Output")


# Case 3 : Discrete input Discrete Output
Tr = np.zeros(25)
Ts = np.zeros(25)
for i in range(num_average_time) :
    x, tr, ts = calc_time('discrete', 'classifier')
    Tr += tr
    Ts += ts
plot_time(x,Tr,Ts,"Discrete input Discrete Output")


# Case 4 : Discrete input Real Output
Tr = np.zeros(25)
Ts = np.zeros(25)
for i in range(num_average_time) :
    x, tr, ts = calc_time('discrete', 'regressor')
    Tr += tr
    Ts += ts
plot_time(x,Tr,Ts,"Discrete input Real Output")