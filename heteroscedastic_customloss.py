# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 09:41:33 2018

@author: bendi
"""
from keras import backend as K
from keras.layers import Input, Dense, Activation, MaxoutDense, Dropout, LeakyReLU
from keras.models import Model, Sequential
from keras import losses, optimizers, backend, regularizers, initializers
import math
import caseloader as cl
from tensorflow import subtract, multiply, exp,matrix_determinant
import tensorflow as tf
import numpy as np
from matplotlib import pyplot

def build_model(neurons, dim, regu, dropout, lr):
    model_1= Sequential()
    model_1.add(Dense(neurons, input_shape=(dim,), kernel_regularizer=regularizers.l2(regu), bias_initializer=initializers.Constant(value=-0.1)))
#    model_1.add(Activation("relu"))
    model_1.add(LeakyReLU(alpha=0.3))
#    model_1.add(Activation("relu"))

    model_1.add(Dropout(dropout))
    model_1.add(Dense(neurons, kernel_regularizer=regularizers.l2(regu), bias_initializer=initializers.Constant(value=0.1)))
    model_1.add(LeakyReLU(alpha=0.3))
#    model_1.add(Activation("relu"))

    model_1.add(Dropout(dropout))
#    model_1.add(Dense(neurons, input_shape=(dim,), kernel_regularizer=regularizers.l2(regu), bias_initializer=initializers.Constant(value=0.1)))
#    model_1.add(Activation("sigmoid"))
#    model_1.add(Dropout(dropout))
#    model_1.add(Dense(neurons, input_shape=(dim,), kernel_regularizer=regularizers.l2(regu), trainable=False))
#    model_1.add(Activation("relu"))
#    model_1.add(Dropout(dropout))
    model_1.add(Dense(2, kernel_regularizer=regularizers.l2(regu)))
    model_1.add(Activation("linear"))
#    model_1.compile(optimizer=optimizers.adam(lr=lr), loss = sced_loss)
    model_1.compile(optimizer=optimizers.SGD(lr=lr), loss=sced_loss)
    return model_1

def sced_loss(y_true, y_pred):
    y_true = K.reshape(y_true, [-1, 1])
    y_pred = K.reshape(y_pred, [-1, 1])
    return K.mean(0.5*multiply(K.exp(-y_pred[1]),K.square((y_pred[0]-y_true[0]))), axis=0)
#    return K.mean(K.exp(-y_pred[1]))
#    return (y_pred[0])


def run(well, separator, runs=10, neurons=3, dim=1, regu=0.0001, dropout=0.05, epochs=1000, batch_size=100, lr=0.1, n_iter=100):
    X, y = cl.BO_load(well, separator)
#    X=2*X
#    y = 550*y
    X_test = [[i] for i in np.arange(np.min(X), np.max(X), (np.max(X)-np.min(X))/n_iter)]
    y = [[i[0], 0] for i in y]
    model = build_model(neurons, dim, regu, dropout, lr)
    print("model built")
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
#    ax2 = fig.add_subplot(111, sharex=ax)
    pyplot.xlabel('gas lift')
    pyplot.ylabel("oil")
    pyplot.show()
#    pyplot.pause(1)
    f = K.function([model.layers[0].input, K.learning_phase()],
                          [model.layers[-1].output])
    for r in range(runs):
        
        #TEST
        inp = model.input                                           
        outputs = [layer.output for layer in model.layers if "activation" in layer.name]
        functor = K.function([inp]+ [K.learning_phase()], outputs )
        updates = opt.get_updates(model.trainable_weights, [], loss, )
        layer_outs = functor([[X[0]], 1.], updates=updates)

        print("prior weights:", model.get_weights()[-2])
        print("net output:", results)
        print("second last layer output:", outputs[-2])
        print("y_true:", y)
        #TEST
#        model.fit([X[0]], [y[0]], batch_size, epochs, verbose=0)
        print("post weights:", model.get_weights()[-2])

        loss = model.evaluate([X[0]], [y[0]])
#        print("run #"+str(r)+" loss:", loss)
        results = np.column_stack(f((X_test,1))[0])
#        print(results)
#        print(np.column_stack(results))
        res_mean = [results[0]]
        res_var = [np.exp(results[1])]
        for i in range(1,n_iter):
            a = np.column_stack(f((X_test,1))[0])
            res_mean.append(a[0])
            res_var.append(np.exp(a[1]))
#            res_mean = np.insert(a[0], [0], a, axis=1)
#            res_var = np.insert(a[1], [0], a, axis=1)
#            results = np.insert(results, [0], a, axis=1)
#        print(res_mean)
        pred_mean = np.mean(res_mean, axis=0)
        pred_sq_mean = np.mean(np.square(res_mean), axis=0)
        var_mean = np.mean(res_var, axis=0)
#        print("\n",pred_sq_mean[:3], pred_mean[:3], var_mean[:3], "\n")
        std = np.sqrt(pred_sq_mean-np.square(pred_mean)+var_mean)
            
#        print("mean var output:", np.mean(var_mean))
        prediction = [x[0] for x in model.predict(X_test)]
        ax.clear()
        ax.plot(X, [i[0] for i in y], linestyle='None', marker = '.',markersize=15)
        ax.plot(X_test,prediction,color='green',linestyle='dashed', linewidth=2)
#        for i in range(2):
#            ax.fill_between([x[0] for x in X_test], pred_mean+std*(i+1)*0.1, pred_mean-std*(i+1)*0.1, alpha=0.2, facecolor='#089FFF', linewidth=2)
        ax.plot(X_test, pred_mean, color='#089FFF', linewidth=1)

        pyplot.pause(0.01)
