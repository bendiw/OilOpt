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
import tools as t
import plotter
from sklearn.preprocessing import normalize, RobustScaler
import random
import math
import copy
import tens
import tensorflow
from keras import losses, optimizers, backend, regularizers, initializers
KERAS_BACKEND = tensorflow

def run(well, separator="HP", epochs = 20000, mode="relu", neurons = 25, goal = 'oil', intervals = 20, factor = 1.5, nan_ratio = 0.3, train_frac = 1.0,
                  val_frac = 0.1, plot = False, save=False, regu = 0.001):
    pyplot.ioff()
    if separator == "HP":
        hp=1
    else:
        hp=0
    #TODO: add regularization, validation etc
# =============================================================================
#     load and normalize data
# =============================================================================
    data = load_well(well, separator, goal, hp, factor, intervals, nan_ratio)
    if (len(data[0][0]) >= 2):
        is_3d = True
        dim = 2
    else:
        is_3d = False
        dim=1
    rs = RobustScaler(with_centering =False)
    if is_3d:
        glift_orig = np.array([x[0][0] for x in data])
        choke_orig = np.array([x[0][1] for x in data])
        y_orig = np.array([x[1][0] for x in data]).reshape(-1,1)
        glift = rs.fit_transform(glift_orig.reshape(-1,1))
        choke = rs.transform(choke_orig.reshape(-1,1))
        y = rs.transform(y_orig.reshape(-2, 1))
        X =[[glift[i][0], choke[i][0]] for i in range(len(glift))]
    else:
        X_orig = np.array([x[0][0] for x in data]).reshape(-1,1)
        y_orig = np.array([x[1][0] for x in data]).reshape(-1,1)
        X = rs.fit_transform(X_orig.reshape(-1,1))
        y = rs.transform(y_orig.reshape(-1, 1))
#    rs.fit(X_orig)
    

# =============================================================================
#     ReLU architecture
# =============================================================================
    if mode=="relu":
            
        model_1= Sequential()
        model_1.add(Dense(neurons, input_shape=(dim,), kernel_regularizer=regularizers.l2(regu)))
        model_1.add(Activation("relu"))
        
        model_1.add(Dense(1, kernel_regularizer=regularizers.l2(regu)))
        model_1.add(Activation("linear"))
# =============================================================================
#     maxout architecture
# =============================================================================
    else:
        a = Input((dim,))
        b = (Dense(neurons, input_shape=(dim,)))(a)
        c = (Dense(neurons, input_shape=(dim,)))(a)
        d = MaxoutDense(output_dim=1)(b)
        e = MaxoutDense(output_dim=1)(c)
        f = Subtract()([d,e])
        model_1 = Model(a, f)
        
    #compile model
    model_1.compile(optimizer=optimizers.adam(lr=0.01), loss = "mean_squared_error")
    
    print(model_1.summary())
# =============================================================================
#     train model on normalized data
# =============================================================================
    model_1.fit(X, y, 
            epochs = epochs, batch_size=100, verbose=0)
    
# =============================================================================
#     initialize new model with pretrained weights
# =============================================================================
    if(mode=="relu"):
        model_2= Sequential()
        model_2.add(Dense(neurons, input_shape=(dim,), weights = [model_1.layers[0].get_weights()[0].reshape(dim,neurons), rs.inverse_transform(model_1.layers[0].get_weights()[1].reshape(-1,1)).reshape(neurons,)]))
        model_2.add(Activation("relu"))
        model_2.add(Dense(1,  weights = [model_1.layers[2].get_weights()[0].reshape(neurons,1), rs.inverse_transform(model_1.layers[2].get_weights()[1].reshape(-1,1)).reshape(1,)]))

    else:
        a_2 = Input((dim,))
        b_2 = (Dense(neurons, input_shape=(dim,), weights = [model_1.layers[1].get_weights()[0].reshape(dim,neurons), rs.inverse_transform(model_1.layers[1].get_weights()[1].reshape(-1,1)).reshape(neurons,)]))(a_2)
        c_2 = (Dense(neurons, input_shape=(dim,), weights = [model_1.layers[2].get_weights()[0].reshape(dim,neurons), rs.inverse_transform(model_1.layers[2].get_weights()[1].reshape(-1,1)).reshape(neurons,)]))(a_2)
        d_2 = MaxoutDense(output_dim=1, weights = [model_1.layers[3].get_weights()[0], rs.inverse_transform(model_1.layers[3].get_weights()[1].reshape(-1,1))])(b_2)
        e_2 = MaxoutDense(output_dim=1, weights = [model_1.layers[4].get_weights()[0], rs.inverse_transform(model_1.layers[4].get_weights()[1].reshape(-1,1))])(c_2)
        f_2 = Subtract()([d_2,e_2])
        model_2 = Model(a_2, f_2)
    model_2.compile(optimizer=optimizers.adam(lr=0.01), loss = "mean_squared_error")

    if save or plot:
        if(is_3d):
            X = rs.inverse_transform(X)
            y = rs.inverse_transform(y)
            prediction = [x for x in model_2.predict(X)]
            fig=plotter.plot3d([x[0] for x in X], [x[1] for x in X], [n[0] for n in prediction] , well)
        else:
            X = rs.inverse_transform(X)
            y = rs.inverse_transform(y)
            prediction = [x for x in model_2.predict(X)]
            fig = pyplot.figure()
            pyplot.plot(X, y, linestyle='None', marker = '.',markersize=8)
            pyplot.plot(X, prediction, color='green', linestyle='dashed')
        if save:
            fig.savefig(well + "-" + separator + "-" + goal+"-fig")
        if plot:
            pyplot.show()
        else:
            pyplot.close(fig)

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
    
def save_all():
    for well in t.wellnames:
        for phase in t.phasenames:
            for sep in t.well_to_sep[well]:
                print(well, phase, sep)
                run(well, separator=sep, goal=phase, save=True, nan_ratio=0)