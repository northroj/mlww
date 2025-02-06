# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 15:47:07 2023

@author: northroj
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import time

def read_data( file_name, num_train):
    data = np.loadtxt(file_name, delimiter=',')
    x_data = data[1:num_train, 1:81]
    y_data = data[1:num_train, 81:]
    x_val = data[num_train:, 1:81]
    y_val = data[num_train:, 81:]
    print("Data loaded")
    return x_data, y_data, x_val, y_val

def normalize_data(y_data, y_val):
    norm_factor1 = np.average(y_data[:, :10])
    norm_factor2 = np.average(y_data[:, 10:20])
    norm_factork = np.average(y_data[:, 20])
    std_1 = np.std(y_data[:,:10])
    std_2 = np.std(y_data[:,10:20])
    std_k = np.std(y_data[:,20])
    
    for i in range(np.size(y_data, 0)):
        y_data[i, :10] = (y_data[i, :10] - norm_factor1) / std_1
        y_data[i, 10:20] = (y_data[i, 10:20] - norm_factor2) / std_2
        y_data[i, 20] = (y_data[i, 20] - norm_factork) / std_k
    for i in range(np.size(y_val, 0)):
        y_val[i, :10] = (y_val[i, :10] - norm_factor1) / std_1
        y_val[i, 10:20] = (y_val[i, 10:20] - norm_factor2) / std_2
        y_val[i, 20] = (y_val[i, 20] - norm_factork) / std_k
    print("Data normalized")

def plot_data( test_case ):
    x_cells = np.arange(1,11,1)
    
    plt.figure(test_case*2)
    plt.plot(x_cells, pred_out[test_case,:10], 'r')
    plt.plot(x_cells, y_val[test_case,:10], 'b')
    plt.xlabel("cell #")
    plt.ylabel('flux')
    plt.title('fast flux')
    plt.legend(["predicted","calculated"])
    plt.show()

    plt.figure(test_case*2+1)
    plt.plot(x_cells,pred_out[test_case,10:20], 'r')
    plt.plot(x_cells, y_val[test_case,10:20], 'b')
    plt.xlabel("cell #")
    plt.ylabel('flux')
    plt.title('thermal flux')
    plt.legend(["predicted","calculated"])
    plt.show()


[x_data,y_data,x_val,y_val] = read_data('testdata2000.csv', 17000)

normalize_data(y_data, y_val)

time1 = time.time()
model1 = MLPRegressor(hidden_layer_sizes=(80, 80, 80), learning_rate_init=0.005, tol=1e-6, activation='logistic', max_iter=2000, alpha=0.0001, shuffle=True, solver='adam').fit(x_data, y_data)
time2 = time.time()
n_epochs = model1.n_iter_
print("Training done in", n_epochs, "epochs, and ", time2-time1, "seconds")

train_r2 = model1.score(x_data, y_data)
val_r2 = model1.score(x_val, y_val)
pred_out = model1.predict(x_val)
kr2 = r2_score(y_val[:,20], pred_out[:,20])
ffluxr2 = r2_score(y_val[:,:10], pred_out[:,:10])
tfluxr2 = r2_score(y_val[:,10:20], pred_out[:,10:20])

plot_data(10)

plot_data(13)


