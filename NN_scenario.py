# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:24:34 2018

@author: arntgm
"""

import pandas as pd
from keras.layers import Input, Dense, Activation, LeakyReLU, PReLU, ELU, MaxoutDense, merge, Subtract, Dropout
from keras.models import Model, Sequential
from keras import losses, optimizers, backend, regularizers, initializers
import numpy as np
import tools
from matplotlib import pyplot
import scipy.stats as ss
# =============================================================================
# This class grossly overfits a PWL NN to variance data from a
# fully trained heteroscedastic NN. Since we do not care about
# generalization properties, no regularization techniques are applied.
# =============================================================================
    

def build_model(neurons, dim, lr, regu=0.0):
    model_1= Sequential()
    model_1.add(Dense(neurons, input_shape=(dim,),
                      kernel_initializer=initializers.VarianceScaling(),
                      bias_initializer=initializers.Constant(value=0.1),
                      kernel_regularizer=regularizers.l2(regu),
                      bias_regularizer=regularizers.l2(regu)))
    model_1.add(Activation("relu"))
    model_1.add(Dense(neurons,
                      kernel_initializer=initializers.VarianceScaling(),
                      bias_initializer=initializers.Constant(value=0.1),
                      kernel_regularizer=regularizers.l2(regu),
                      bias_regularizer=regularizers.l2(regu)))
    model_1.add(Activation("relu"))
    model_1.add(Dense(neurons,
                      kernel_initializer=initializers.VarianceScaling(),
                      bias_initializer=initializers.Constant(value=0.1),
                      kernel_regularizer=regularizers.l2(regu),
                      bias_regularizer=regularizers.l2(regu)))
    model_1.add(Activation("relu"))
#    model_1.add(Dense(neurons*2,
#                      kernel_initializer=initializers.VarianceScaling(),
#                      bias_initializer=initializers.Constant(value=0.1),
#                      kernel_regularizer=regularizers.l2(regu),
#                      bias_regularizer=regularizers.l2(regu)))
#    model_1.add(Activation("relu"))

    model_1.add(Dense(1,
                      kernel_initializer=initializers.VarianceScaling(),
                      bias_initializer=initializers.Constant(value=0.1)))
    model_1.add(Activation("linear"))
    model_1.compile(optimizer=optimizers.Adam(lr=lr), loss="mse")
    return model_1

# =============================================================================
# main function
# =============================================================================
def train_scen(well, goal='oil', neurons=15, dim=1, case=2, lr=0.005, batch_size=50,
        epochs=1000, save=False, plot=False, num_std=4, regu=0.0):
    filename = "variance_case"+str(case)+"_"+goal+".csv"
    df = pd.read_csv(filename, sep=';', index_col=0)
    for w in well:
        model = build_model(neurons, dim, lr, regu=regu)
        mean = df[str(w)+"_"+goal+"_mean"]
        std = df[str(w)+"_"+goal+"_var"]
        X = np.array([[i] for i in range(len(mean))])
        y = np.zeros(len(mean))
        for i in range(len(mean)):
            y[i] = np.max([ss.truncnorm.rvs(-num_std, num_std, scale=std[i], loc=mean[i], size=(1)),0])
        model.fit(X,y,batch_size=batch_size,epochs=epochs,verbose=0)
        if plot or save:
            prediction = [x[0] for x in model.predict(X)]
            fig = pyplot.figure()
            ax = fig.add_subplot(111)
            line1 = ax.plot(X, y, linestyle='None', marker = '.',markersize=5)
            line2 = ax.plot(X, prediction, color='green',linestyle='dashed', linewidth=1)
            pyplot.xlabel('choke')
            pyplot.ylabel(goal)
            if save:
                filepath = "scenarios\\nn\\"+str(w)+"_"+goal+"_(2-"+str(neurons)+","+str(epochs)+","+str(lr)+")"+".png"
                pyplot.savefig(filepath, bbox_inches="tight")
                tools.save_variables(w+"_(2-"+str(neurons)+","+str(epochs)+","+str(lr)+")", goal=goal, case=2,neural=model.get_weights(), mode="scen", folder="scenarios\\nn\\")
            if plot:
                pyplot.show()


def train_all_scen(neurons=15,lr=0.005,epochs=1000,save=True,plot=False, case=2, num_std=4):
    for w in t.wellnames_2:
        for p in ["oil","gas"]:
            train_scen(w, goal=p, neurons=neurons, lr=lr, epochs=epochs, save=save, plot=plot, case=case, num_std=num_std)

            
    
#    model = retrieve_model(dims, w, b)
#    X_sample = np.array([[i] for i in range(101)])
#    y_sample = np.array([x[0] for x in model.predict(X_sample)])
#    scen_model = build_model(neurons, dim, lr)
#    scen_model.fit(X_sample, y_sample, batch_size, epochs, verbose=0)
#    if save:
#        t.save_variables(well, goal=goal, neural=scen_model.get_weights(), mode="var", case=case, folder="scenario\nn\\")
#    if plot:
#        prediction = [x[0] for x in scen_model.predict(X_sample)]
#        fig = pyplot.figure()
#        ax = fig.add_subplot(111)
#        line1 = ax.plot(X_sample, y_sample, linestyle='None', marker = '.',markersize=10)
#        line2 = ax.plot(X_sample, prediction, color='green',linestyle='dashed', linewidth=1)
#        pyplot.xlabel('choke')
#        pyplot.ylabel(goal)
#        pyplot.show()