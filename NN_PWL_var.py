# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 10:50:19 2018

@author: bendiw
"""
import pandas as pd
from keras.layers import Input, Dense, Activation, LeakyReLU, PReLU, ELU, MaxoutDense, merge, Subtract, Dropout
from keras.models import Model, Sequential
from keras import losses, optimizers, backend, regularizers, initializers
import numpy as np
from matplotlib import pyplot
# =============================================================================
# This class grossly overfits a PWL NN to variance data from a
# fully trained heteroscedastic NN. Since we do not care about
# generalization properties, no regularization techniques are applied.
# =============================================================================


# =============================================================================
# build a ReLU NN
# =============================================================================
def build_model(neurons, dim, regu, lr):
    model_1= Sequential()
    model_1.add(Dense(neurons, input_shape=(dim,),
                      kernel_initializer=initializers.VarianceScaling(),
#                      kernel_regularizer=regularizers.l2(regu), 
#                      bias_regularizer=regularizers.l2(regu),
                      bias_initializer=initializers.Constant(value=0.1)))
    model_1.add(Activation("relu"))
#    

    model_1.add(Dense(neurons, 
                      kernel_initializer=initializers.VarianceScaling(),
#                      kernel_regularizer=regularizers.l2(regu),
#                      bias_regularizer=regularizers.l2(regu),
                      bias_initializer=initializers.Constant(value=0.1)))
    model_1.add(Activation("relu"))

    model_1.add(Dense(1,
                      kernel_initializer=initializers.VarianceScaling(),
#                      kernel_regularizer=regularizers.l2(regu),
#                      bias_regularizer=regularizers.l2(regu),
                      bias_initializer=initializers.Constant(value=0.1)))
    model_1.add(Activation("linear"))
    model_1.compile(optimizer=optimizers.Adam(lr=lr), loss="mse")
    return model_1


# =============================================================================
# main function
# =============================================================================
def run(well, goal, neurons, dim, lr=0.0001, batch_size=128, epochs=1000):
     model = build_model(neurons, dim, lr)
     filename = well+"_"+goal+"var.csv"
     data = pd.read_csv(filename, sep=';', index_col=0)
     X = data[well+"_"+goal+"_X"]
     y = data[well+"_"+goal+"std"]
     model.fit(X, y, batch_size, epochs, verbose=0)
     prediction = [x[0] for x in model.predict(X)]
     fig = pyplot.figure()
     ax = fig.add_subplot(111)
     line1 = ax.plot(X, [i[0] for i in prediction], linestyle='None', marker = '.',markersize=10)
     pyplot.xlabel('choke')
     pyplot.ylabel(goal)
     pyplot.show()
