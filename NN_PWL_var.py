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
import tools as t
from matplotlib import pyplot
from sklearn.preprocessing import RobustScaler
# =============================================================================
# This class grossly overfits a PWL NN to variance data from a
# fully trained heteroscedastic NN. Since we do not care about
# generalization properties, no regularization techniques are applied.
# =============================================================================


# =============================================================================
# build a ReLU NN
# =============================================================================
def build_model(neurons, dim, lr, extra_layer=False):
    model_1= Sequential()
    model_1.add(Dense(neurons, input_shape=(dim,),
                      kernel_initializer=initializers.VarianceScaling(),
#                      kernel_regularizer=regularizers.l2(regu), 
#                      bias_regularizer=regularizers.l2(regu),
                      bias_initializer=initializers.Constant(value=0.1)))
    model_1.add(Activation("relu"))
#    

#    model_1.add(Dense(neurons, 
#                      kernel_initializer=initializers.VarianceScaling(),
##                      kernel_regularizer=regularizers.l2(regu),
##                      bias_regularizer=regularizers.l2(regu),
#                      bias_initializer=initializers.Constant(value=0.1)))
#    model_1.add(Activation("relu"))

    model_1.add(Dense(1,
                      kernel_initializer=initializers.VarianceScaling(),
#                      kernel_regularizer=regularizers.l2(regu),
#                      bias_regularizer=regularizers.l2(regu),
                      bias_initializer=initializers.Constant(value=0.1)))
    model_1.add(Activation("linear"))
    if extra_layer:
        model_1.add(Dense(1,
                      weights = [np.array([[1000000]]), np.array([0])]))
    model_1.compile(optimizer=optimizers.Adam(lr=lr), loss="mse")
    return model_1

# =============================================================================
# main function
# =============================================================================
def run(well, goal='oil', neurons=40, dim=1, case=2, lr=0.001, batch_size=50,
        epochs=10000, save=False, plot=True):
     model = build_model(neurons=neurons, dim=dim, lr=lr)
     filename = "variance_case"+str(case)+"_"+goal+".csv"
     data = pd.read_csv(filename, sep=';', index_col=0)
     X = data[well+"_"+goal+"_X"]
     y = data[well+"_"+goal+"_var"]
#     y_ = (y**2)/1000000.0
     y_ = y**2
     if goal == "gas":
         y_ = y_/1000000.0
#     X = rs.transform(X.reshape(-1,1))
     model.fit(X, y_, batch_size, epochs, verbose=0)
     prediction = [x[0] for x in model.predict(X)]
     if goal == "gas":
         model_2 = t.add_layer(model, neurons, "mse")
         prediction_2 = [x[0] for x in model_2.predict(X)]
     else:
        model_2 = model
     if save:
         t.save_variables(well, goal=goal, neural=model_2.get_weights(), mode="var", case=case)
     if plot:
         fig = pyplot.figure()
         ax = fig.add_subplot(111)
         line1 = ax.plot(X, y_, linestyle='None', marker = '.',markersize=10)
         line2 = ax.plot(X, prediction, color='green',linestyle='dashed', linewidth=1)
         pyplot.xlabel('choke')
         pyplot.ylabel(goal)
         pyplot.show()
         if goal == "gas":
             fig = pyplot.figure()
             ax = fig.add_subplot(111)
             line1 = ax.plot(X, y_*1000000.0, linestyle='None', marker = '.',markersize=10)
             line2 = ax.plot(X, prediction_2, color='green',linestyle='dashed', linewidth=1)
             pyplot.xlabel('choke2')
             pyplot.ylabel(goal)
             pyplot.show()
  

