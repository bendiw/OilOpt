# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:00:28 2018

@author: bendi
"""

from sklearn.model_selection import GridSearchCV
from BOestimator import NetEstimator
import tools as t
import caseloader as cl
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Input, Dense, Activation, MaxoutDense, Dropout
from keras.models import Model, Sequential
from keras import losses, optimizers, backend, regularizers, initializers
from scipy.special import logsumexp
from keras import backend as K


def log_likelihood(tau, N):
    def ll(y_true, y_pred):
#        print(y_true.eval())
        return (K.logsumexp(-0.5 * tau * (y_true - y_pred)**2., 0) - 0.5*K.log(tau))
#        return logsumexp(-0.5 * tau * (y_true - y_pred)**2., 0) - np.log(N) - 0.5*np.log(2*np.pi) + 0.5*np.log(tau)
    return ll

def create_model(tau=0.005, dim=1, length_scale=0.001, dropout=0.05, score="ll", 
                 mode="relu", neurons = 25, learn_rate = 0.1, N=1000):
    #regularization parameter is calc based on hyperparameters
    regu = length_scale**2 * (1 - dropout) / (2. * N * tau)
    model_1 = None
    #ReLU
    
#    print("tau:", tau, "\tlength_scale:", length_scale, "\tdropout:", dropout)
    if mode=="relu":
        model_1= Sequential()
        model_1.add(Dense(neurons, input_shape=(dim,), kernel_regularizer=regularizers.l2(regu)))
        model_1.add(Activation("relu"))
        model_1.add(Dropout(dropout))
        model_1.add(Dense(1, kernel_regularizer=regularizers.l2(regu)))
        model_1.add(Activation("linear"))
    
    #maxout
    elif mode == "maxout":
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
    else:
        raise ValueError('mode/architecture not supported!')
    #compile model
    loglik = log_likelihood(tau, N)
    model_1.compile(optimizer=optimizers.adam(lr=learn_rate), loss = "mean_squared_error", metrics=[loglik])
    return model_1


def search(well, separator, parameters=t.param_dict):
    X, y = cl.BO_load(well, separator)
    parameters['N'] = [len(X)]
    model = KerasRegressor(build_fn=create_model, epochs = 600, batch_size=128, verbose=0)
    gs = GridSearchCV(model, parameters, verbose=2)
    gs.fit(X, y)
    print(gs.best_params_)