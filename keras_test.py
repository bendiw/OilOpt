# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:28:18 2018

@author: bendiw
"""
from keras.layers import Input, Lambda, Dense, Activation, LeakyReLU, PReLU, ELU
from keras.models import Model, Sequential
import numpy as np
from matplotlib import pyplot
import caseloader as cl
import tools
import plotter
from sklearn.preprocessing import normalize
import random
import math
import copy
import tens
from keras import losses, optimizers, backend, regularizers, initializers

def run(epochs = 1000, neurons = 15, goal = 'oil', intervals = 20, factor = 1.5, nan_ratio = 0.3, hp = 1, train_frac = 0.8,
                  val_frac = 0.1, plot = True):
    
    model_1= Sequential()
    model_1.add(Dense(neurons, input_shape=(1,)))
    model_1.add(Activation("relu"))
    
#    model_1.add(ELU())
    
#    model_1.add(Dense(neurons,))
#    model_1.add(Activation("relu"))
    model_1.add(Dense(1, ))
    model_1.add(Activation("linear"))
#    model_1.add(Lambda(lambda x: backend.sum(x, axis=0),output_shape=(1,)))
    model_1.compile(optimizer=optimizers.adam(), loss = "mean_squared_error")
# =============================================================================

    
    deg = 3
#    all_data_orig= [[i, random.randint(0, i+5*(i+1))] for i in range(10)]
#    all_data_orig = [[x, x**deg] for x in np.arange(-3, 3, 0.5)]
#    for i in range(0,len(all_data_orig), 2):
#        all_data_orig[i][1] = random.randint(-i, i)
    all_data_orig = [[94218.22902686402, 202.88931920759566], [98382.84899564099, 137.42681397671845], [100147.06876999052, 36.34664144212609], [103971.12043888686, 82.04293442285885], [104921.42497957515, 147.50283790283785], [119435.29586281738, 46.55084437315633], [120773.1523822956, 71.94729306017095], [122231.47520490436, 56.97724788683303], [124718.30296345525, 47.911507710447616]]
#    
#    all_data_orig = normalize(all_data_orig, axis=0)
#    print(all_data_orig)

    all_data_orig = pos_norm(all_data_orig)
    
    
    all_data = copy.deepcopy(all_data_orig)
    print(all_data)
    random.shuffle(all_data)
    train_set = all_data[0:math.floor(len(all_data)*train_frac)]
    validation_set = all_data[len(train_set):len(all_data)]
    model_1.fit(np.array([x[0] for x in train_set]), np.array([x[1] for x in train_set]), 
                epochs = epochs, batch_size=20, verbose=0)
    print(model_1.summary())
#    weights = model_1.layers[0].get_weights()[0]
#    biases = model_1.layers[0].get_weights()[1]
#    print("w:", weights)
#    print("\n b:", biases)
    total_x = [[x[0]] for x in all_data_orig]
    total_y = [[x[1]] for x in all_data_orig]
#    print(total_x)
    prediction = [x for x in model_1.predict(total_x)]
#    for i in range(len(prediction)):
#        print("x:", total_x[i], "pred:", prediction[i])
    pyplot.figure()
    pyplot.plot(total_x, total_y, linestyle='None', marker = '.',markersize=8)
    pyplot.plot(total_x, prediction, color='green', linestyle='dashed')

def pos_norm(data):
    max_in = 0
    max_out = 0
    min_in = 0
    min_out = 0
    for i in data:
        max_in = max(max_in, i[0])
        min_in = min(min_in, i[0])
        max_out = max(max_out, i[1])
        min_out = min(min_out, i[1])    
    print(min_out)
    return [[(data[i][0]-min_in)/(max_in-min_in), (data[i][1]-min_out)/(max_out-min_out)] for i in range(len(data))]