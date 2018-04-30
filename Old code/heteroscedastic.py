# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 13:03:26 2018

@author: bendi
"""
from keras import backend as K
from keras.layers import Input, Dense, Activation, MaxoutDense, Dropout
from keras.models import Model, Sequential
from keras import losses, optimizers, backend, regularizers, initializers
import math
import caseloader as cl
import numpy as np
from matplotlib import pyplot


def build_model(neurons, dim, regu, dropout, lr):
    model_1= Sequential()
    model_1.add(Dense(neurons, input_shape=(dim,), kernel_regularizer=regularizers.l2(regu), trainable=False))
    model_1.add(Activation("relu"))
    model_1.add(Dropout(dropout))
    model_1.add(Dense(neurons, input_shape=(dim,), kernel_regularizer=regularizers.l2(regu), trainable=False))
    model_1.add(Activation("relu"))
    model_1.add(Dropout(dropout))
    model_1.add(Dense(neurons, input_shape=(dim,), kernel_regularizer=regularizers.l2(regu), trainable=False))
    model_1.add(Activation("relu"))
    model_1.add(Dropout(dropout))
    model_1.add(Dense(neurons, input_shape=(dim,), kernel_regularizer=regularizers.l2(regu), trainable=False))
    model_1.add(Activation("relu"))
    model_1.add(Dropout(dropout))
    model_1.add(Dense(2, kernel_regularizer=regularizers.l2(regu), trainable=False))
    model_1.add(Activation("linear"))
    model_1.compile(optimizer=optimizers.adam(lr=lr), loss = mean_squared_error)
    return model_1

def sced_loss(y_true, y_pred):
#    print(outputs)
#    return np.zeros_like(outputs)
#    return 0.5*(outputs[0]-y_true)*(outputs[0]-y_true)*math.exp(-outputs[1])
    z = K.mean(y_true-y_pred, axis=-1)
    print(z)
    return K.mean(y_true-y_pred, axis=-1)

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)
    
def mse_loss(y_true, outputs):
    print("output tensors:", outputs)
    return 0.5*(outputs[0]-y_true)**2

def calc_grads(model, loss, layer_outs, y_true):
    old_w = model.get_weights()
    het_grad, prop_grad = hetero_sced_grad(old_w[-2:], loss, layer_outs[-2:], y_true)
#    print(old_w)
    rel_grad = relu_grad(old_w[:-2], layer_outs[:-1], prop_grad)
    grad = np.append(rel_grad, het_grad)
    return grad
    
# =============================================================================
# calculates gradients in last layer
# =============================================================================
def hetero_sced_grad(weights, loss, layer_outs, y_true):
    grad = np.zeros((len(weights[0]),2))   #zero out gradient
    prec = math.exp(-layer_outs[1][0][0])
    dy = (layer_outs[0][0][0]-y_true)
    for in_neuron in range(len(weights[0])):
        grad[in_neuron][0] += prec*dy
        grad[in_neuron][1] -= 0.5*(prec*dy*dy-1)
    loc_grad = np.multiply(grad, layer_outs[0][0][:, np.newaxis]) #gradient wrt parameters
    het_grad = [loc_grad, np.append(prec*dy, -0.5*(prec*dy*dy-1))] #list of weight updates, bias updates
    prop_grad = np.sum(np.multiply(grad,weights[0]), axis=1) #gradient wrt input, propagates
    return het_grad, prop_grad
    
# =============================================================================
# calculates gradient in intermittent layers
# =============================================================================
def relu_grad(weights, layer_outs, prop_grad):
    rel_grad = []
#    print("prop", prop_grad)
    for l in range(len(layer_outs)-1,-1,-1): #iterate over layers
        if(l==0): #first layer
            rel_grad.append(prop_grad) #add bias
            grad = np.zeros_like(layer_outs[l])
            loc_grad = weights[0]*prop_grad
            rel_grad.append(loc_grad) #add gradient
        else:
            rel_grad.append(prop_grad) #add bias
#            print("prop:", prop_grad)
            grad = np.zeros((len(layer_outs[l-1][0]), len(layer_outs[l][0])))
#            print("zero grad:", grad)
            old_prop = np.copy(prop_grad)
#            print("weights prev:", weights[l*2])
            prop_grad = np.sum(weights[l*2]*prop_grad, axis=1)
#            print("Vdw:", prop_grad)
            loc_grad = np.matmul(layer_outs[l-1].T,old_prop[np.newaxis,:])
#            print("tfi.dw:", loc_grad)
            rel_grad.append(loc_grad) #add gradient
    rel_grad.reverse()
    return rel_grad
        

def fit(model, X, y, epochs, batch_size, lr, plot=True, neurons=8):
    inp = model.input                                           
    outputs = [layer.output for layer in model.layers if "activation" in layer.name]

    functor = K.function([inp]+ [K.learning_phase()], outputs )
    momentum = 0.9
#    prev_grad = np.zeros(len(model.get_weights()))
    prev_grad = [0 for x in range(len(model.get_weights()))]
    for e in range(epochs):
        losses_sced = np.zeros((batch_size+1,))
        losses_mse = np.zeros((batch_size+1,))
        idx = np.random.choice(np.arange(len(X)), batch_size+1, replace=True)
        X_batch = X[idx]
        y_batch = y[idx]
        for b in range(batch_size):
            layer_outs = functor([[X_batch[b]], 1.])
            loss_sced = sced_loss(y_batch[b], layer_outs[-1][0])
            grad = calc_grads(model, loss_sced, layer_outs, y[0])
#            print("gradients:", grad)
#            print("\n\nweights:", model.get_weights())
            if(b==0):
                grads=grad
            else:
                grads+= grad
            b += 1
#        print(grads)
            losses_sced[b] = loss_sced
            losses_mse[b] = mse_loss(y_batch[b], layer_outs[-1][0])
        gradient = grads/batch_size
        weights = model.get_weights()
        for j in range(len(grad)):
            prev_grad[j] = momentum*prev_grad[j] -lr*grad[j]
            weights[j]+= prev_grad[j]
#        prev_grad = gradient
        model.set_weights(weights)
        print("Epoch:", e, "\nMSE loss:", np.mean(losses_mse), "\nScedastic loss:", np.mean(losses_sced), "\n")
        print("outputs:", layer_outs[-1])
    if(plot):
        steps = 50
        step_size = float((X.max()-X.min())/float(steps))
        mean_input = [[i] for i in np.arange((np.round(X.min()*0.5)),
                      (np.round(X.max()*1.3))+step_size, step_size)]
#        mean, var = get_mean_var(model, 0.05, 0.0001, mean_input,10, 100)

#        prediction = [[x[0], math.exp(x[1])] for x in model.predict(X)]
        prediction = [x[0] for x in model.predict(X)]
        fig = pyplot.figure()
        pyplot.plot(X, y, linestyle='None', marker = '.',markersize=15)
#        pyplot.plot([x[0] for x in mean_input], mean, color='#089FFF')
#        pyplot.fill_between([x[0] for x in mean_input], mean-np.power(var,0.5),
#                            mean+np.power(var,0.5),
#                           alpha=0.2, facecolor='#089FFF', linewidth=1)
#        pyplot.fill_between([x[0] for x in mean_input], mean-0.5*np.power(var,0.5),
#                            mean+0.5*np.power(var,0.5),
#                           alpha=0.2, facecolor='#089FFF', linewidth=2)                   
        pyplot.plot(X,prediction,color='green',linestyle='dashed', linewidth=3)
        pyplot.xlabel('gas lift')
        pyplot.ylabel("oil")
        pyplot.show()
    
    
def run(well, separator, neurons=3, dim=1, regu=0.0001, dropout=0.05, epochs=1000, batch_size=100, lr=0.1):
    X, y = cl.BO_load(well, separator)
    print(X, y)
    model = build_model(neurons, dim, regu, dropout, lr)
    fit(model, X, y, epochs, batch_size, lr, neurons=neurons)
    
    # =============================================================================
# returns the mean and the variation of a data sample X, as a consequence of the
# dropout regularization
# =============================================================================
def get_mean_var(model, dropout, regu, X, length_size, iterations):
    f = backend.function([model.layers[0].input, backend.learning_phase()],
                          [model.layers[-1].output])
    t = tau(regu, length_size, dropout, len(X))
    return predict_uncertainty(f, X, iterations, t)

def predict_uncertainty(f, x, n_iter, t):
    results = f((x,1))[0]
    for i in range(1,n_iter):
        a = f((x,1))[0]
        results = np.insert(results, [0], a, axis=1)
    pred_mean = np.mean(results, axis=1)
    pred_var = np.var(results, axis=1) + math.pow(t, -1)
    return pred_mean, pred_var

def tau(regu, length_rate, dropout, datapoints):
    tau = math.pow(length_rate,2) * (1-dropout)
    return tau/float(2*datapoints*regu)