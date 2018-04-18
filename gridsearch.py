# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:00:28 2018

@author: bendi
"""

from sklearn.model_selection import GridSearchCV
import tools as t
import caseloader as cl
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Input, Dense, Activation, MaxoutDense, Dropout
from keras.models import Model, Sequential
from keras import losses, optimizers, backend, regularizers, initializers
from scipy.special import logsumexp
from keras import backend as K
import numpy as np
import types
import tempfile
import keras.models
import pickle


# =============================================================================
# Wrapper class for Keras net to use in sklearn's grid search
# =============================================================================
class NeuralRegressor(KerasRegressor):
        def score(self, x, y, **kwargs):
            """Returns the mean loss on the given test data and labels.
            # Arguments
                x: array-like, shape `(n_samples, n_features)`
                    Test samples where `n_samples` is the number of samples
                    and `n_features` is the number of features.
                y: array-like, shape `(n_samples,)`
                    True labels for `x`.
                **kwargs: dictionary arguments
                    Legal arguments are the arguments of `Sequential.evaluate`.
            # Returns
                score: float
                    Mean accuracy of predictions on `x` wrt. `y`.
            """
            kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)
            loss = self.model.evaluate(x, y, **kwargs)
            print(self.model.metrics_names)
            print(loss)
            if isinstance(loss, list):
                return loss[-1]
            return -loss

def log_likelihood(tau, N):
    def ll(y_true, y_pred):
#        print(y_true.eval())
        return K.mean((K.logsumexp(-0.5 * tau * (y_true - y_pred)**2., 0) + 0.5*K.log(tau)))
#        return logsumexp(-0.5 * tau * (y_true - y_pred)**2., 0) - np.log(N) - 0.5*np.log(2*np.pi) + 0.5*np.log(tau)
    return ll


def sced_loss(y_true, y_pred):
    return K.mean(0.5*y_pred[1] + 0.5*multiply(K.exp(-y_pred[1]),K.square(y_pred[0]-y_true[0])), axis=0) # 

def create_model(tau=0.005, length_scale=0.001, dropout=0.05, score="ll", 
                 mode="relu", neurons = 25, learn_rate = 0.1, N=1000, variance="homosced"):
    #regularization parameter is calc based on hyperparameters
    regu = length_scale**2 * (1 - dropout) / (2. * N * tau)
    model_1 = None
    #ReLU
    
#    print("tau:", tau, "\tlength_scale:", length_scale, "\tdropout:", dropout)
    if mode=="relu":
        model_1= Sequential()
        model_1.add(Dense(neurons, input_shape=(dim,),
                      kernel_initializer=initializers.VarianceScaling(),
                      kernel_regularizer=regularizers.l2(regu), 
                      bias_initializer=initializers.Constant(value=0.1),
                      bias_regularizer=regularizers.l2(regu)))
        model_1.add(Activation("relu"))
        model_1.add(Dropout(dropout))
        model_1.add(Dense(neurons,
                      kernel_initializer=initializers.VarianceScaling(),
                      kernel_regularizer=regularizers.l2(regu), 
                      bias_initializer=initializers.Constant(value=0.1),
                      bias_regularizer=regularizers.l2(regu)))
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
    if(variance=="homosced"):
        loss="mse"
    else:
        loss=sced_loss
    model_1.compile(optimizer=optimizers.adam(lr=learn_rate), loss = loss, metrics=[loglik])
    return model_1



def search(well, separator, parameters=t.param_dict):
    make_keras_picklable()
    X, y = cl.BO_load(well, separator)
    global dim
<<<<<<< HEAD
    dim = len(X[0])
    parameters['N'] = [len(X)]
=======
    global N
    dim = len(X[0])
    N = len(X)
>>>>>>> master
    model = NeuralRegressor(build_fn=create_model, epochs = 600, batch_size=128, verbose=0)
    gs = GridSearchCV(model, parameters, verbose=2)
    gs_result = gs.fit(X, y)
    print(gs.best_score_, gs.best_params_)
    print(gs_result)