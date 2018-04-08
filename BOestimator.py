# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 11:27:08 2018

@author: bendi
"""

from sklearn.base import BaseEstimator
from keras.layers import Input, Dense, Activation, MaxoutDense, Dropout
from keras.models import Model, Sequential
from keras import losses, optimizers, backend, regularizers, initializers

class NetEstimator:

    def __init__(self, verbose=0, tau=0.005, dim=1, length_scale=0.001, dropout=0.05, score="ll", epochs = 20000, batch_size=128, mode="relu", neurons = 25, learn_rate = 0.1, N=1000):
        
        #regularization parameter is calc based on hyperparameters
        print(length_scale)
        regu = length_scale**2 * (1 - dropout) / (2. * N * tau)
        model_1 = None
        self.score=score
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose=verbose
        self.tau = tau
        self.N = N
        #ReLU
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
        self.net = model_1.compile(optimizer=optimizers.adam(lr=learn_rate), loss = "mean_squared_error")

    

    def score(self, X, y=None):
        print("score")
        if self.score =="ll":
            self.predict(X, y)
            return self.ll
        elif self.score == "mse":
            losses = self.net.evaluate(X, y)
            return np.mean(losses)
        
        
    def fit(self, X, y=None):
        print("fit")
        hist = self.net.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return hist
    
    def predict(self, X, y=None):
        print("predict")
        y_pred = self.net.predict(X, y)
        
        #calc log likelihood
        self.ll = (logsumexp(-0.5 * self.tau * (y - y_pred)**2., 0) - np.log(self.N) 
            - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
#        self.ll = np.mean(test_ll)
        return y_pred