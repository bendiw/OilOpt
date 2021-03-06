# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:00:28 2018

@author: bendi
"""

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import tools as t
import caseloader as cl
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Input, Dense, Activation, MaxoutDense, Dropout
from keras.models import Model, Sequential
from keras import losses, optimizers, backend, regularizers, initializers
from scipy.special import logsumexp
from keras import backend as K
from tensorflow import multiply
import math
import numpy as np
import types
import pandas as pd
import tempfile
import keras.models
import pickle
from datetime import datetime

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
#            print(self.model.metrics_names)
#            print(loss)
            if isinstance(loss, list):
                return loss[-1]
            return -loss

def log_likelihood_homosced(tau, N):
    def ll(y_true, y_pred):
#        print(y_true.eval())
        return K.mean((K.logsumexp(-0.5 * tau * (y_true - y_pred)**2., 0) + 0.5*K.log(tau)) - K.log(N)-0.5*K.log(2.*math.pi))
#        return logsumexp(-0.5 * tau * (y_true - y_pred)**2., 0) - np.log(N) - 0.5*np.log(2*np.pi) + 0.5*np.log(tau)
    return ll

def log_likelihood_heterosced(N):
    def ll(y_true, y_pred):
        return K.mean(K.logsumexp(-0.5 * K.exp(-y_pred[:,1]) * (y_true[:,0] - y_pred[:,0])**2., 0) + 0.5*K.log(K.exp(-y_pred[:,1])) - K.log(N)-0.5*K.log(2.*math.pi))
    return ll


def sced_loss(y_true, y_pred):
    return K.mean(0.5*multiply(K.exp(-y_pred[:,1]),K.square(y_pred[:,0]-y_true[:,0]))+0.5*y_pred[:,1], axis=0)

def create_model(tau=0.005, length_scale=0.001, dropout=0.05, score="ll", 
                 mode="relu", neurons = 25, learn_rate = 0.001, N=1000., variance="heterosced", regu=0.0001, layers=2):
    #regularization parameter is calc based on hyperparameters
    if(variance!="homosced"):
        tau=1.
        output_dim = 2
    else:
        output_dim = 1
#    regu = length_scale**2 * (1 - dropout) / (2. * N * tau)
    model_1 = None
    #ReLU
    
#    print("tau:", tau, "\tlength_scale:", length_scale, "\tdropout:", dropout)
    if mode=="relu":
        model_1= Sequential()
        for i in range(layers):
            model_1.add(Dense(neurons, input_shape=(dim,),
                          kernel_initializer=initializers.VarianceScaling(),
                          kernel_regularizer=regularizers.l2(regu), 
                          bias_initializer=initializers.Constant(value=0.1),
                          bias_regularizer=regularizers.l2(regu)))
            model_1.add(Activation("relu"))
            model_1.add(Dropout(dropout))
        model_1.add(Dense(output_dim, kernel_regularizer=regularizers.l2(regu)))
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
    if(variance=="homosced"):
        loss="mse"
        loglik = log_likelihood_homosced(tau, N)
    else:
        loss=sced_loss
        loglik = log_likelihood_heterosced(N)
    model_1.compile(optimizer=optimizers.adam(lr=learn_rate), loss = loss, metrics=[loglik])
    return model_1



def search(well, separator="HP", case=1, parameters=t.param_dict, variance="heterosced", x_grid=None, y_grid=None, 
           verbose=2, nan_ratio=0.0, goal='gas', mode="grid", n_iter=30, epochs=5000, expanded=False):
    if(well):
        X, y,_ = cl.BO_load(well, separator, case=case, nan_ratio=nan_ratio, goal=goal)

        if(x_grid is not None):
            print("Datapoints before merge:",len(X))
        X=np.array(X)
        y=np.array(y)
        if(x_grid is not None):
            X,y = t.simple_node_merge(np.array(X),np.array(y),x_grid,y_grid)
            print("Datapoints after merge:",len(X))
    else:
        #generate simple test data
        #sine curve with some added faulty data
        X = np.arange(-3., 4., 7/40)
        y = [[math.sin(x), 0] for x in X]
        X = np.append(X, [1.,1.,1.,1.,1.])
        y.extend([[1.,0.],[2.,0.],[0.5,0.],[0.,0.], [1.7,0.]])
    global dim
    global N
    dim = len(X[0])
    N = float(len(X))

    model = NeuralRegressor(build_fn=create_model, epochs = epochs, batch_size=20, verbose=0)
    
    if(mode=="grid"):
        if(expanded):
            parameters = t.param_dict_expanded
        else:
            parameters = t.param_dict
        gs = GridSearchCV(model, parameters, verbose=verbose, return_train_score=True, error_score=np.NaN)
    else:
        if(expanded):
#            raise ValueError("Expanded randomized parameters not specified yet!")
            parameters = t.param_dict_rand_expanded
        else:
            parameters = t.param_dict_rand
        gs = RandomizedSearchCV(model, parameters, verbose=verbose, return_train_score=True, n_iter=n_iter, error_score=np.NaN)
        
    gs.fit(X, y)
    grid_result = gs.cv_results_
    df = pd.DataFrame.from_dict(grid_result)
    filestring = "gridsearch/"+well+"_"+goal+("_"+separator if case==1 else "")+("_"+mode)+("x_"+str(x_grid) if x_grid is not None else "")+("_expanded" if expanded else "")+".csv"
    with open(filestring, 'w') as f:
        df.to_csv(f, sep=';', index=False)
    print("Best: %f using %s" % (gs.best_score_, gs.best_params_))

    if(verbose==0):
        means = grid_result['mean_test_score']
        stds = grid_result['std_test_score']
        params = grid_result['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

def search_all(case=2, goal="gas", mode="grid", n_iter=90, x_grid=None, y_grid=None):
    if(case==2):
        for w in t.wellnames_2[5:]:
            print(w, goal, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            search(w, case=case, verbose=0, goal=goal, mode=mode, n_iter=n_iter, x_grid=x_grid, y_grid=y_grid)
    else:
        for w in t.wellnames[-2:]:
            for sep in t.well_to_sep[w]:
                for goal in ["oil","gas"]:
                    print(w, sep, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    search(w, separator=sep, verbose=0, goal=goal)
