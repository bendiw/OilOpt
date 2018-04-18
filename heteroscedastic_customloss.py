# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 09:41:33 2018

@author: bendi
"""
from keras import backend as K

from keras.layers import Dense, Activation, Dropout, LeakyReLU
from keras.models import Sequential
from keras import optimizers, regularizers, initializers
import math
import caseloader as cl
from tensorflow import multiply
import numpy as np
from matplotlib import pyplot
import tools


# =============================================================================
# builds a neural net
# will probably want to expand to let call determine architecture
# =============================================================================
def build_model(neurons, dim, regu, dropout, lr):
    model_1= Sequential()

    model_1.add(Dense(neurons, input_shape=(dim,),
                      kernel_initializer=initializers.VarianceScaling(),
                      kernel_regularizer=regularizers.l2(regu), 
                      bias_initializer=initializers.Constant(value=0.1),
                      bias_regularizer=regularizers.l2(regu)))
    model_1.add(Activation("relu"))
#    model_1.add(LeakyReLU(alpha=0.3))
    model_1.add(Dropout(dropout))
#    
#    model_1.add(Dense(neurons, input_shape=(dim,),
#                      kernel_initializer=initializers.VarianceScaling(),
#                      kernel_regularizer=regularizers.l2(regu), 
#                      bias_initializer=initializers.Constant(value=-0.1),
#                      bias_regularizer=regularizers.l2(regu)))
#    model_1.add(Activation("relu"))
#    model_1.add(Dropout(dropout))

    model_1.add(Dense(neurons, 
                      kernel_initializer=initializers.VarianceScaling(),
                      kernel_regularizer=regularizers.l2(regu), 
                      bias_initializer=initializers.Constant(value=0.1),
                      bias_regularizer=regularizers.l2(regu)))
    model_1.add(Activation("relu"))
#    model_1.add(LeakyReLU(alpha=0.3))
    model_1.add(Dropout(dropout))

    model_1.add(Dense(2, kernel_regularizer=regularizers.l2(regu)))
    model_1.add(Activation("linear"))
    model_1.compile(optimizer=optimizers.Adamax(lr=lr), loss=sced_loss)
    return model_1

# =============================================================================
# Heteroscedastic loss function. See Yarin Gal
# =============================================================================
def sced_loss(y_true, y_pred):
    return K.mean(0.5*y_pred[1] + 0.5*multiply(K.exp(-y_pred[1]),K.square(y_pred[0]-y_true[0])), axis=0) #  
    
def mse_loss(y_true, y_pred):
    return K.mean(K.square(y_pred[0]-y_true[0]))



# =============================================================================
# load data, create a model and train it
# =============================================================================
def run(well=None, separator="HP", x_grid=None, y_grid=None, case=1, runs=10, neurons=3, dim=1, regu=0.0001, dropout=0.05, epochs=1000, batch_size=100, lr=0.1, n_iter=100, scaler="rs"):
    if(well):
        X, y = cl.BO_load(well=well, separator=separator, scaler=scaler, case=case)
        if(x_grid and y_grid):
            print("Datapoints before merge:",len(X))
            X=np.array(X)
            y=np.array(y)
            X,y = tools.simple_node_merge(np.array(X),np.array(y),x_grid,y_grid)
            print("Datapoints after merge:",len(X))
    else:
        #generate simple test data
        #sine curve with some added faulty data
        X = np.arange(-3., 4., 7/40)
        y = [[math.sin(x), 0] for x in X]
        X = np.append(X, [1.,1.,1.,1.])
        y.extend([[1.,0.],[2.,0.],[0.5,0.], [1.7,0.]])

    step = (np.max(X)-np.min(X))/n_iter
    X_test = np.array([[i] for i in np.arange(np.min(X), np.max(X)*1.2+step, step)]) #-np.max(X)*0.2
    y = np.array([[i[0], 0] for i in y])
    model = build_model(neurons, dim, regu, dropout, lr)


    #setup plots
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    pyplot.xlim(np.min(X)-0.2*np.max(X), np.max(X)+0.2*np.max(X))
    pyplot.ylim(np.min([i[0] for i in y])-0.05*np.max([i[0] for i in y]), np.max(y)+0.05*np.max([i[0] for i in y]))

    pyplot.autoscale(False)
    pyplot.xlabel('choke')
    pyplot.ylabel("oil")
    pyplot.show()
    
    #forward pass function, needed for dropout eval
    f = K.function([model.layers[0].input, K.learning_phase()],
                          [model.layers[-1].output])
    for r in range(runs):
        #train model
        model.fit(X, y, batch_size, epochs, verbose=0)
#        if r==0:
#            model.fit(X, y, batch_size, epochs, verbose=0)
#        else:
#            rate = lr/(1+np.log10(r))
#            K.set_value(model.optimizer.lr, rate)
#            model.fit(X,y,batch_size,epochs,verbose=0)
#        print(K.get_value(model.optimizer.lr))


        #gather results from forward pass
        results = np.column_stack(f((X_test,1.))[0])
        if (np.isnan(results[0][0])):
            print("NAN")
            return

        res_mean = [results[0]]
        res_var = [np.exp(results[1])]
        for i in range(1,n_iter):
            a = np.column_stack(f((X_test,1.))[0])
            res_mean.append(a[0])
            res_var.append(np.exp(a[1]))
        
        #calculate uncertainty
        pred_mean = np.mean(res_mean, axis=0)
        pred_sq_mean = np.mean(np.square(res_mean), axis=0)
        var_mean = np.mean(res_var, axis=0)
        std = np.sqrt(pred_sq_mean-np.square(pred_mean)+var_mean)
        
        #this is the current network's prediction with dropout switched off
        prediction = [x[0] for x in model.predict(X_test)]
        
        #plot results from current run
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
        pyplot.pause(0.001)
        
        
        
        
# =============================================================================
#         Run a forward pass through the model
#        and record output values and gradients, weights, biases.
#        useful for testing
# =============================================================================
def test_gradients(model, X, y):
    #format of input
    input_tensors = [model.inputs[0], # input data
         model.sample_weights[0], # how much to weight each sample by
         model.targets[0], # labels
         K.learning_phase(), # train or test mode
         ]
    
    #sample parameters
    inputs = [[X[0]], # X
          [1], # weight single sample fully
          [y[0]], # y sample
          1 # training, i.e. doprout is active
      ]
    #setup functions
    inp = model.input                                           
    weights = model.trainable_weights
    layer_outs = [layer.output for layer in model.layers if "activation" in layer.name or "leaky" in layer.name]
    gradients = model.optimizer.get_gradients(model.total_loss, weights)
    get_gradients = K.function(inputs=input_tensors, outputs=gradients)
    functor = K.function([inp]+ [K.learning_phase()], layer_outs )
    
    #run forward pass(es)
    loss = model.evaluate(np.array([X[0]]), np.array([y[0]]))
    layer_outputs = functor([[X[0]], 1.])
    grads = get_gradients(inputs)
    
    #print whatever you're interested in here
    print ("gradients:",grads[-2:])
    return
