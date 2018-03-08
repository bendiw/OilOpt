# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:28:18 2018

@author: bendiw
"""
from keras.layers import Input, Lambda, Dense, Activation, LeakyReLU, PReLU, ELU, MaxoutDense, merge, Subtract, Dropout
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
from keras import losses, optimizers, backend, regularizers, initializers

def run(well, separator="HP", epochs = 20000, mode="relu", neurons = 25, goal = 'oil', intervals = 20, factor = 1.5, nan_ratio = 0.3, train_frac = 1.0,
                  val_frac = 0.1, plot = False, save=False, regu = 0.001, dropout = 0.2, lr = 0.01):
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
#        model_1.add(Dropout(dropout, input_shape=(dim,)))
        model_1.add(Dense(neurons, input_shape=(dim,), kernel_regularizer=regularizers.l2(regu)))
        model_1.add(Activation("relu"))
        model_1.add(Dropout(dropout))
        model_1.add(Dense(1, kernel_regularizer=regularizers.l2(regu)))
        model_1.add(Activation("linear"))
# =============================================================================
#     maxout architecture
# =============================================================================
    else:
        a = Input((dim,))
        b = Dropout(dropout, input_shape=(dim,))(a)
        c = Dense(neurons)(b)
        d = Dense(neurons)(b)
        e = Dropout(dropout)(c)
        f = Dropout(dropout)(d)
        g = MaxoutDense(output_dim=1)(e)
        h = MaxoutDense(output_dim=1)(f)
        i = Subtract()([g,h])
        model_1 = Model(a, i)
        
    #compile model
    model_1.compile(optimizer=optimizers.adam(lr=lr), loss = "mean_squared_error")
    
    print(model_1.summary())
# =============================================================================
#     train model on normalized data
# =============================================================================
    model_1.fit(X, y, 
            epochs = epochs, batch_size=100, verbose=0)
    
# =============================================================================
#     initialize new model with pretrained weights
# =============================================================================
    if mode == "relu":        
        model_2= Sequential()
#        model_2.add(Dropout(dropout,input_shape=(dim,)))
        model_2.add(Dense(neurons, input_shape=(dim,), weights = [model_1.layers[0].get_weights()[0].reshape(dim,neurons),
                          rs.inverse_transform(model_1.layers[0].get_weights()[1].reshape(-1,1)).reshape(neurons,)], kernel_regularizer=regularizers.l2(regu)))
        model_2.add(Activation("relu"))
        model_2.add(Dropout(dropout))
        model_2.add(Dense(1,  weights = [model_1.layers[3].get_weights()[0].reshape(neurons,1),rs.inverse_transform(model_1.layers[3].get_weights()[1].reshape(-1,1)).reshape(1,)],
                                         kernel_regularizer=regularizers.l2(regu)))
        model_2.compile(optimizer=optimizers.adam(lr=lr), loss = "mean_squared_error")

#    else:
        

    X = rs.inverse_transform(X)
    y = rs.inverse_transform(y)
    if save or plot:
        if(is_3d):
            prediction = [x for x in model_2.predict(X)]
            plotter.plot3d([x[0] for x in X], [x[1] for x in X], [n[0] for n in prediction] , well)
        else:
            mean_input = [[i] for i in range(int(np.round(X.min())),
                          int(np.round(X.max()))+1, int(np.round((X.max()-X.min())/1000.0)))]
            mean, var = get_mean_var(model_2, dropout, regu, mean_input, 0.1, 50)
            prediction = [x for x in model_2.predict(X)]
            fig = pyplot.figure()
            pyplot.plot(X, y, linestyle='None', marker = '.',markersize=8)
            pyplot.plot([x[0] for x in mean_input], mean, color='#089FFF')
            pyplot.fill_between([x[0] for x in mean_input], mean-2*np.power(var,0.5),
                                mean+2*np.power(var,0.5),
                               alpha=0.2, facecolor='#089FFF', linewidth=1)
            pyplot.plot(X,prediction,color='green',linestyle='dashed')
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
        

# =============================================================================
# returns the mean and the variation of a data sample X, as a consequence of the
# dropout regularization
# =============================================================================
def get_mean_var(model, dropout, regu, X, length_size, iterations):
    f = backend.function([model.layers[0].input, backend.learning_phase()],
                          [model.layers[-1].output])
    t = tau(regu, length_size, dropout, len(X))
    return predict_uncertainty(f, X, iterations, t)

def tau(regu, length_rate, dropout, datapoints):
    tau = math.pow(length_rate,2) * (1-dropout)
    return tau/float(2*datapoints*regu)

def predict_uncertainty(f, x, n_iter, t):
    results = f((x,1))[0]
    for i in range(1,n_iter):
        a = f((x,1))[0]
        results = np.insert(results, [0], a, axis=1)
    pred_mean = np.mean(results, axis=1)
    pred_var = np.var(results, axis=1) + math.pow(t, -1)
    return pred_mean, pred_var

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