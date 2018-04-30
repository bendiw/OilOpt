# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:28:18 2018

@author: bendiw
"""
from keras.layers import Input, Dense, Activation, LeakyReLU, PReLU, ELU, MaxoutDense, merge, Subtract, Dropout
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

def run(well, case = 2, separator="HP", epochs = 5000, mode="relu", neurons = 5, goal = 'oil', intervals = 20, factor = 1.5, nan_ratio = 0.3, train_frac = 1.0,
                  val_frac = 0.1, plot = False, save=False, regu = 0.001, dropout = 0.2, lr = 0.01, length_scale = 10):
    pyplot.ioff()
    if separator == "HP":
        hp=1
    else:
        hp=0
    #TODO: add regularization, validation etc
# =============================================================================
#     load and normalize data
# =============================================================================
    X,y = load_well(well, separator, goal, hp, factor, intervals, nan_ratio,case)
#    data = [[[x], [math.sin(x)*x**3]] for x in np.arange(0, 9, 1.)]
#    data.append([[8.2], [30]])
#    data.append([[8.5], [-2]])
#    data = [[[140000],[120]], [[142000],[130]],[[148000],[140]],[[158000],[150]],[[151000],[155]]]
#    for i in data:
#        i[0][0]=i[0][0]/5000.0
#    print (data)
#    data.sort()

    if (case == 1 and len(X[0]) >= 2):
        is_3d = True
        dim = 2
    else:
        is_3d = False
        dim=1

    rs = RobustScaler(with_centering =False)
    if is_3d:
        glift_orig = np.array([x[0] for x in X])
        choke_orig = np.array([x[0] for x in X])
        y_orig = np.array([x[0] for x in y]).reshape(-1,1)
        glift = rs.fit_transform(glift_orig.reshape(-1,1))
        choke = rs.transform(choke_orig.reshape(-1,1))
        y = rs.transform(y_orig.reshape(-2, 1))
        X =np.array([[glift[i][0], choke[i][0]] for i in range(len(glift))])
    else:
        X = rs.fit_transform(X)
        y = rs.transform(y)
#    rs.fit(X_orig)
    

# =============================================================================
#     ReLU architecture
# =============================================================================
    if mode=="relu":
            
        model_1= Sequential()
#        model_1.add(Dropout(dropout, input_shape=(dim,)))
        model_1.add(Dense(neurons, input_shape=(dim,), kernel_regularizer=regularizers.l2(regu), bias_initializer=initializers.Constant(0.1)))
        model_1.add(Activation("relu"))
        model_1.add(Dropout(dropout))
# =============================================================================
        model_1.add(Dense(int(neurons), kernel_regularizer=regularizers.l2(regu), bias_initializer=initializers.Constant(0.1)))
        model_1.add(Activation("relu"))
        model_1.add(Dropout(dropout))
#        model_1.add(Dense(int(neurons), kernel_regularizer=regularizers.l2(regu)))
#        model_1.add(Activation("relu"))
#        model_1.add(Dropout(dropout))
#        model_1.add(Dense(int(neurons), kernel_regularizer=regularizers.l2(regu)))
#        model_1.add(Activation("relu"))
#        model_1.add(Dropout(dropout))

#        model_1.add(Dense(int(neurons), kernel_regularizer=regularizers.l2(regu)))
#        model_1.add(Activation("relu"))
#        model_1.add(Dropout(dropout))
# =============================================================================
        model_1.add(Dense(1, kernel_regularizer=regularizers.l2(regu), bias_initializer=initializers.Constant(value=0.1)))
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
    model_1.compile(optimizer=optimizers.adam(lr=lr,decay=0.001), loss = "mean_squared_error")
#    model_1.compile(loss="mean_squared_error", optimizer="sgd")
    
    print(model_1.summary())

# =============================================================================
#     train model on normalized data
# =============================================================================
    model_1.fit(X, y, 
            epochs = epochs, batch_size=100, verbose=0)
    print("fitted")

    
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
# =============================================================================
        model_2.add(Dense(neurons, weights = [model_1.layers[3].get_weights()[0].reshape(neurons,neurons),
                          rs.inverse_transform(model_1.layers[3].get_weights()[1].reshape(-1,1)).reshape(neurons,)], kernel_regularizer=regularizers.l2(regu)))
        model_2.add(Activation("relu"))
        model_2.add(Dropout(dropout))
#        model_2.add(Dense(neurons, weights = [model_1.layers[6].get_weights()[0].reshape(neurons,neurons),
#                          rs.inverse_transform(model_1.layers[6].get_weights()[1].reshape(-1,1)).reshape(neurons,)], kernel_regularizer=regularizers.l2(regu)))
#        model_2.add(Activation("relu"))
#        model_2.add(Dropout(dropout))
#        model_2.add(Dense(neurons, weights = [model_1.layers[9].get_weights()[0].reshape(neurons,neurons),
#                          rs.inverse_transform(model_1.layers[9].get_weights()[1].reshape(-1,1)).reshape(neurons,)], kernel_regularizer=regularizers.l2(regu)))
#        model_2.add(Activation("relu"))
#        model_2.add(Dropout(dropout))
#        model_2.add(Dense(neurons, weights = [model_1.layers[12].get_weights()[0].reshape(neurons,neurons),
#                          rs.inverse_transform(model_1.layers[12].get_weights()[1].reshape(-1,1)).reshape(neurons,)], kernel_regularizer=regularizers.l2(regu)))
#        model_2.add(Activation("relu"))
#        model_2.add(Dropout(dropout))
# =============================================================================
        model_2.add(Dense(1,  weights = [model_1.layers[-2].get_weights()[0].reshape(neurons,1),rs.inverse_transform(model_1.layers[-2].get_weights()[1].reshape(-1,1)).reshape(1,)],
                                         kernel_regularizer=regularizers.l2(regu)))
        model_2.add(Activation("linear"))
        model_2.compile(optimizer=optimizers.adam(lr=lr), loss = "mean_squared_error")

#    else:
        

    X = rs.inverse_transform(X)
    y = rs.inverse_transform(y)
    if save or plot:
        if(is_3d):
            prediction = [x for x in model_2.predict(X)]
            fig=plotter.plot3d([x[0] for x in X], [x[1] for x in X], [n[0] for n in prediction] , well)
        else:
            steps = 50
            step_size = float((X.max()-X.min())/float(steps))
            mean_input = [[i] for i in np.arange((np.round(X.min()*0.9)),
                          (np.round(X.max()*1.05))+step_size, step_size)]
            pred_input = np.array([[i] for i in np.arange((np.round(X.min())),
                          (np.round(X.max()))+step_size, step_size)])
            mean, var = get_mean_var(model_2, dropout, regu, mean_input, length_scale, 100)

            prediction = [x for x in model_2.predict(pred_input)]
            fig = pyplot.figure()
            pyplot.plot(X, y, linestyle='None', marker = '.',markersize=15)
            pyplot.plot([x[0] for x in mean_input], mean, color='#089FFF')
            pyplot.fill_between([x[0] for x in mean_input], mean-np.power(var,0.5),
                                mean+np.power(var,0.5),
                               alpha=0.2, facecolor='#089FFF', linewidth=1)
            pyplot.fill_between([x[0] for x in mean_input], mean-0.5*np.power(var,0.5),
                                mean+0.5*np.power(var,0.5),
                               alpha=0.2, facecolor='#089FFF', linewidth=2)                   
            pyplot.plot(pred_input,prediction,color='green',linestyle='dashed', linewidth=3)
            pyplot.xlabel('gas lift')
            pyplot.ylabel(goal)
            pyplot.title(well+"\nlength scale: "+ str(length_scale) + ", dropout: "+str(dropout))
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
        t.save_variables(well, hp, goal, is_3d, model_2.get_weights())
        

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
    r = f((x,1))
    results = r[0]
    for i in range(1,n_iter):
        a = f((x,1))[0]
        results = np.insert(results, [0], a, axis=1)
    pred_mean = np.mean(results, axis=1)
    pred_var = np.var(results, axis=1) + math.pow(t, -1)
    return pred_mean, pred_var

def load_well(well, separator, goal, hp, factor, intervals, nan_ratio, case):
#    df = cl.load("welltests_new.csv")
#    dict_data,_,_ = cl.gen_targets(df, well+"", goal=goal, normalize=False, intervals=intervals,
#                               factor = factor, nan_ratio = nan_ratio, hp=hp) #,intervals=100
#    data = tens.convert_from_dict_to_tflists(dict_data)
    X,y = cl.BO_load(well, case = case, separator = separator, goal = goal)
    return X,y


    
def save_all():
    for well in t.wellnames:
        for phase in t.phasenames:
            for sep in t.well_to_sep[well]:
                print(well, phase, sep)
                run(well, separator=sep, goal=phase, save=True, nan_ratio=0)