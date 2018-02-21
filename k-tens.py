# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 08:36:16 2018

@author: agmal_000
"""

from keras.layers import Input, Lambda, Dense
from keras.models import Model
import numpy as np
import random as r
from matplotlib import pyplot
import caseloader as cl
import tools
import plotter
import math
import tens
from keras import losses, optimizers, backend, regularizers, initializers

def add_layer(input_layer, name, neurons, activation = 'relu'):
    layer = Dense(neurons, activation = activation, name = name, kernel_initializer=initializers.random_uniform(minval=-0.5,maxval=0.5),
                  kernel_regularizer=regularizers.l2(0.5), bias_regularizer=regularizers.l2(0.1), activity_regularizer=regularizers.l2(0.1))(input_layer)
    return layer

def build_models(neurons = 40):
    inputs_1 = Input(shape=(1,))
    inputs_2 = Input(shape=(2,))
    
    layer_1 = add_layer(inputs_1, "relu_1D", neurons)
    layer_2 = add_layer(inputs_2, "relu_2D", neurons)
    
    output_1 = Lambda(lambda x: backend.sum(x), output_shape = (1,))(layer_1)
    output_2 = Lambda(lambda x: backend.sum(x), output_shape = (1,))(layer_2)
    
    model_1 = Model(inputs = inputs_1, outputs = output_1)
    model_2 = Model(inputs = inputs_2, outputs = output_2)
    Model.get_output_shape_at()
        
    model_1.compile(optimizer=optimizers.adam(lr=0.01), loss = losses.mean_squared_error)
    model_2.compile(optimizer=optimizers.adam(lr=0.01), loss = losses.mean_squared_error)
    return model_1, model_2


    
def train_on_well(datafile, models, goal = 'oil', intervals = 20, factor = 1.5, nan_ratio = 0.3, hp = 1, train_frac = 0.8,
                  val_frac = 0.1, plot = False):
    df = cl.load("welltests_new.csv")
    dict_data, means, stds = cl.gen_targets(df, datafile+"", goal=goal, normalize=False, intervals=intervals,
                               factor = factor, nan_ratio = nan_ratio, hp=hp) #,intervals=100
    data = tens.convert_from_dict_to_tflists(dict_data)
    print(data)
    is_3d = False
    if (len(data[0][0]) >= 2):
        model = models[1]
        is_3d = True
        print("Well",datafile, goal, "- Choke and gaslift")
    else:
        model = models[0]
        print("Well",datafile, goal, "- Gaslift only")
        
    train_set, validation_set, test_set = tens.generate_sets(data, train_frac, val_frac)
    model.fit(np.array([x[0] for x in train_set]), np.array([x[1] for x in train_set]), epochs = 1000, batch_size=20, verbose=1)
    grid_size=10
    if (is_3d):
        x_vals = tens.get_x_vals(dict_data, grid_size)
        prediction = model.predict(np.array(x_vals))
        print (prediction)
        y_vals = [[0] for i in range(len(x_vals))]
#        print(x_vals)
#        x_vals, y_vals, prediction = tens.denormalize(x_vals, y_vals, [[x] for x in prediction], means, stds)
        x1 = [x[0] for x in x_vals]
        x2 = [x[1] for x in x_vals]
        z = []
        for pred in prediction:
            z.append(pred)
        if (plot):
            plotter.plot3d(x1, x2, z, datafile)
        breakpoints = tools.delaunay(x1,x2,z) 
    else:
#        total_x, total_y, pred = denormalize(total_x, total_y, pred, means, stds)
        xvalues, yvalues = [], []
        for i in range(len(total_x)):
            xvalues.append(total_x[i][0])
            yvalues.append(pred[i][0])
        breakpoints = [[xvalues[0],yvalues[0]]]
        breakpoints_y = [yvalues[0]]
        breakpoints_x = [xvalues[0]]
        old_w = (yvalues[1]-yvalues[0])/(xvalues[1]-xvalues[0])
        for i in range(2,len(yvalues)):
            w = (yvalues[i]-yvalues[i-1])/(xvalues[i]-xvalues[i-1])
            if (abs(w-old_w)>0.00001):
                breakpoints.append([xvalues[i-1],yvalues[i-1]])
                breakpoints_y.append(yvalues[i-1])
                breakpoints_x.append(xvalues[i-1])
            old_w = w
        breakpoints.append([xvalues[-1],yvalues[-1]])
        breakpoints_y.append(yvalues[-1])
        breakpoints_x.append(xvalues[-1])
        if (plot):
            plot_pred(total_x, pred, total_y)
            pyplot.ylabel(goal)
            pyplot.xlabel('gas lift')
            pyplot.title(datafile)
            pyplot.plot(breakpoints_x, breakpoints_y, 'k*')
            pyplot.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    