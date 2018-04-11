# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 09:41:33 2018

@author: bendi
"""
from keras import backend as K
from keras.layers import Input, Dense, Activation, MaxoutDense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras import losses, optimizers, backend, regularizers, initializers
import math
import caseloader as cl
from tensorflow import subtract, multiply, exp,matrix_determinant
import tensorflow as tf
import numpy as np
from matplotlib import pyplot
import tools


def build_model(neurons, dim, regu, dropout, lr):
    model_1= Sequential()
    model_1.add(Dense(neurons, input_shape=(dim,), 
                      kernel_regularizer=regularizers.l2(regu),
                      kernel_initializer=initializers.RandomNormal(mean=0.0,stddev=0.1),
                      bias_initializer=initializers.Constant(value=0.1),
                      bias_regularizer=regularizers.l2(regu)))
#    model_1.add(LeakyReLU(alpha=0.3))
    model_1.add(Activation("relu"))

    model_1.add(Dropout(dropout))
    model_1.add(Dense(neurons, kernel_regularizer=regularizers.l2(regu),
                      kernel_initializer=initializers.RandomNormal(mean=0.0,stddev=0.1),
                      bias_initializer=initializers.Constant(value=0.1),
                      bias_regularizer=regularizers.l2(regu)))
#    model_1.add(LeakyReLU(alpha=0.3))
    model_1.add(Activation("relu"))

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
    model_1.compile(optimizer=optimizers.Adagrad(lr=lr), loss=sced_loss)
    return model_1

def sced_loss(y_true, y_pred):
#    y_true = K.reshape(y_true, [-1, 1])
#    y_pred = K.reshape(y_pred, [-1, 1])
    return K.mean(0.5*multiply(K.exp(-y_pred[1]),K.square(y_pred[0]-y_true[0])) + 0.5*y_pred[1], axis=0)
#    return K.mean(K.exp(-y_pred[1]))
#    return (y_pred[0])
    
def mse_loss(y_true, y_pred):
    return K.mean(K.square(y_pred[0]-y_true[0]))


def run(well, separator, case=2, runs=200, x_grid=40, y_grid=40, neurons=20,
        dim=1, regu=0.00001, dropout=0.05, epochs=10, batch_size=100, lr=0.05, n_iter=100):
    X, y = cl.BO_load(well, separator)
    X = np.array(X)
    y = np.array(y)
    print("Datapoints before merge:",len(X))
    X,y = tools.simple_node_merge(X,y,x_grid,y_grid)
    print("Datapoints after merge:",len(X))
#    X=2*X
#    y = 550*y
    step = (np.max(X)-np.min(X))/n_iter
    X_test = np.array([[i] for i in np.arange(np.min(X)-0.2*np.max(X), np.max(X)*1.2+step, step)])
#    y = [[i[0], 0] for i in y]
    model = build_model(neurons, dim, regu, dropout, lr)
    print("model built")
    
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    pyplot.xlim(np.min(X)-0.1*np.max(X), np.max(X)+0.2*np.max(X))
    pyplot.ylim(np.min([i[0] for i in y])-0.25*np.max([i[0] for i in y]), 1.25*np.max([i[0] for i in y]))
    pyplot.autoscale(False)
    pyplot.xlabel('gas lift')
    pyplot.ylabel("oil")
    pyplot.show()
#    pyplot.pause(1)
    
    #TEST
#    inp = model.input                                           
#    weights = model.trainable_weights
##        print([n.name for n in model.layers])
#    layer_outs = [layer.output for layer in model.layers if "activation" in layer.name or "leaky" in layer.name]
#    gradients = model.optimizer.get_gradients(model.total_loss, weights)
#    input_tensors = [model.inputs[0], # input data
#             model.sample_weights[0], # how much to weight each sample by
#             model.targets[0], # labels
#             K.learning_phase(), # train or test mode
#             ]
#    get_gradients = K.function(inputs=input_tensors, outputs=gradients)
#    functor = K.function([inp]+ [K.learning_phase()], layer_outs )
    #TEST
#    model.fit(X, y, batch_size, 130)
    f = K.function([model.layers[0].input, K.learning_phase()],
                          [model.layers[-1].output])
    for r in range(runs):
        
        #TEST
#        layer_outputs = functor([[X[0]], 1.])
#        print("prior weights:", model.get_weights()[-2])
#        print("tzzT:", layer_outputs)
        inputs = [[X[0]], # X
                  [1], # sample weights
                  [y[0]], # y
                  1 # learning phase in TEST mode
              ]

#        print("net output:", layer_outputs[-1])
#        print("second last layer output:", layer_outputs[-2])
#        print ("gradients:",get_gradients(inputs)[-2:])
#        print("y_true:", y[0])
        #TEST

        model.fit(X, y, batch_size, epochs, verbose=0)
#        if r==0:
#            model.fit(X, y, batch_size, epochs, verbose=0)
#        else:
#            rate = lr/(1+np.log10(r))
#            K.set_value(model.optimizer.lr, rate)
#            model.fit(X,y,batch_size,epochs,verbose=0)
#        print(K.get_value(model.optimizer.lr))


#        loss = model.evaluate(np.array([X[0]]), np.array([y[0]]))
#        print("run #"+str(r)+" loss:", loss)
        results = np.column_stack(f((X_test,1))[0])
        if (np.isnan(results[0][0])):
            print("NAN")
            return
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
#        ax.clear()
        if r==0:
            line1 = ax.plot(X, [i[0] for i in y], linestyle='None', marker = '.',markersize=15)
            line2 = ax.plot(X_test,prediction,color='green',linestyle='dashed', linewidth=2)
            for i in range(2):
                (ax.fill_between([x[0] for x in X_test], pred_mean+std*(i+1), pred_mean-std*(i+1), alpha=0.2, facecolor='#089FFF', linewidth=2))
            line3 = ax.plot(X_test, pred_mean, color='#089FFF', linewidth=1)
        else:
            line2[0].set_ydata(prediction)
            line3[0].set_ydata(pred_mean)
            pyplot.draw()
            ax.collections.clear()
            for i in range(2):
                (ax.fill_between([x[0] for x in X_test], pred_mean+std*(i+1), pred_mean-std*(i+1), alpha=0.2, facecolor='#089FFF', linewidth=2))

            
        pyplot.pause(0.01)
