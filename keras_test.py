# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:28:18 2018

@author: bendiw
"""
from keras.layers import Input, Lambda, Dense, Activation, LeakyReLU, PReLU, ELU, MaxoutDense, merge, Subtract
from keras.models import Model, Sequential
import numpy as np
from matplotlib import pyplot
import caseloader as cl
import tools
import plotter
from sklearn.preprocessing import normalize, RobustScaler
import random
import math
import copy
import tens
from keras import losses, optimizers, backend, regularizers, initializers

def run(well, separator="HP", epochs = 1000, mode="relu", neurons = 5, goal = 'oil', intervals = 20, factor = 1.5, nan_ratio = 0.3, hp = 1, train_frac = 1.0,
                  val_frac = 0.1, plot = True, save=False):
    
    
    #TODO: add regularization, validation etc
    
# =============================================================================
#     ReLU architecture
# =============================================================================
    if mode=="relu":
            
        model_1= Sequential()
        model_1.add(Dense(neurons, input_shape=(1,)))
        model_1.add(Activation("relu"))
        
        model_1.add(Dense(1, ))
        model_1.add(Activation("linear"))
# =============================================================================
#     maxout architecture
# =============================================================================
    else:
        a = Input((1,))
        b = (Dense(neurons, input_shape=(1,)))(a)
        c = (Dense(neurons, input_shape=(1,)))(a)
        d = MaxoutDense(output_dim=1)(b)
        e = MaxoutDense(output_dim=1)(c)
        f = Subtract()([d,e])
        model_1 = Model(a, f)
        
    #compile model
    model_1.compile(optimizer=optimizers.adam(lr=0.01), loss = "mean_squared_error")
    
    
# =============================================================================
#     load and normalize data
# =============================================================================
    data = load_well(well, separator, goal, hp, factor, intervals, nan_ratio)
    if (len(data[0][0]) >= 2):
        is_3d = True
    else:
        is_3d = False
        
    rs = RobustScaler(with_centering =False)
    if(not is_3d):
        X_orig = np.array([x[0][0] for x in data]).reshape(-1,1)
        y_orig = np.array([x[1][0] for x in data]).reshape(-1,1)
    rs.fit(X_orig)
    X = rs.fit_transform(X_orig.reshape(-1,1))
    y = rs.transform(y_orig.reshape(-1, 1))
    
    
# =============================================================================
#     train model on normalized data
# =============================================================================
    model_1.fit(X, y, 
            epochs = epochs, batch_size=100, verbose=0)
    
    if plot:
        prediction = [x for x in model_1.predict(X)]
        pyplot.figure()
        pyplot.plot(X, y, linestyle='None', marker = '.',markersize=8)
        pyplot.plot(X, prediction, color='green', linestyle='dashed')
        
# =============================================================================
#     initialize new model with pretrained weights
# =============================================================================
    model_2= Sequential()
    model_2.add(Dense(neurons, input_shape=(1,), weights = [model_1.layers[0].get_weights()[0].reshape(1,neurons), rs.inverse_transform(model_1.layers[0].get_weights()[1].reshape(-1,1)).reshape(neurons,)]))
    model_2.add(Activation("relu"))
    model_2.add(Dense(1,  weights = [model_1.layers[2].get_weights()[0].reshape(neurons,1), rs.inverse_transform(model_1.layers[2].get_weights()[1].reshape(-1,1)).reshape(1,)]))
    model_2.compile(optimizer=optimizers.adam(lr=0.01), loss = "mean_squared_error")

    if plot:
        X = rs.inverse_transform(X)
        y = rs.inverse_transform(y)
        prediction = [x for x in model_2.predict(X)]
        pyplot.figure()
        pyplot.plot(X, y, linestyle='None', marker = '.',markersize=8)
        pyplot.plot(X, prediction, color='green', linestyle='dashed')


# =============================================================================
#     save denormalized model to file
# =============================================================================
    if save:
        save_variables(well, hp, goal, is_3d, model_2.get_weights())

def load_well(well, separator, goal, hp, factor, intervals, nan_ratio):
    df = cl.load("welltests_new.csv")
    dict_data,_,_ = cl.gen_targets(df, well+"", goal=goal, normalize=False, intervals=intervals,
                               factor = factor, nan_ratio = nan_ratio, hp=hp) #,intervals=100
    data = tens.convert_from_dict_to_tflists(dict_data)
    return data

def save_variables(datafile, hp, goal, is_3d, neural):
    for l in neural:
        print("\n", l)
    if(hp==1):
        sep = "HP"
    else:
        sep = "LP"
    filename = "" + datafile + "-" + sep + "-" + goal
#    print("Filename:", filename)
    file = open(filename + ".txt", "w")
    if (is_3d):
        file.write("2\n")
    else:
        file.write("1\n")
        for i in range(0,3,2):
            line = ""
            w = neural[i]
            for x in w:
                for y in x:
                    line += str(y) + " "
            file.write(line+"\n")
        for i in range(1,4,2):
            line = ""
            b = neural[i]
            for x in b:
                line += str(x) + " "
            file.write(line+"\n")
    file.close()