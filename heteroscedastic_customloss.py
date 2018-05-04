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
import tools, plotter
import pandas as pd


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

    model_1.add(Dense(neurons, 
                      kernel_initializer=initializers.VarianceScaling(),
                      kernel_regularizer=regularizers.l2(regu), 
                      bias_initializer=initializers.Constant(value=0.1),
                      bias_regularizer=regularizers.l2(regu)))
    model_1.add(Activation("relu"))
#    model_1.add(LeakyReLU(alpha=0.3))
    model_1.add(Dropout(dropout))

    model_1.add(Dense(2,
                      kernel_initializer=initializers.VarianceScaling(),
                      kernel_regularizer=regularizers.l2(regu),
                      bias_initializer=initializers.Constant(value=0.1),
                      bias_regularizer=regularizers.l2(regu)))
    model_1.add(Activation("linear"))
    model_1.compile(optimizer=optimizers.Adam(lr=lr), loss=sced_loss)
    return model_1

# =============================================================================
# Heteroscedastic loss function. See Yarin Gal
# =============================================================================
def sced_loss(y_true, y_pred):
    return K.mean(0.5*multiply(K.exp(-y_pred[:,1]),K.square(y_pred[:,0]-y_true[:,0]))+0.5*y_pred[:,1], axis=0)
    
def mse_loss(y_true, y_pred):
    return K.mean(0.5*K.square(y_pred[0]-y_true[0]))



# =============================================================================
# load data, create a model and train it
# =============================================================================
def run(well=None, separator="HP", x_grid=None, y_grid=None, case=1, runs=10,
        neurons=20, dim=1, regu=0.00001, dropout=0.05, epochs=1000,
        batch_size=50, lr=0.001, n_iter=50, sampling_density=50, scaler='rs',
        goal="oil", save_variance = False, save_weights = False):
    if(well):
        X, y, rs = cl.BO_load(well, separator, case=case, scaler=scaler, goal=goal)
        if(x_grid is not None and case==2):
            print("Datapoints before merge:",len(X))
        if(x_grid is not None and case==2):
            X,y = tools.simple_node_merge(np.array(X),np.array(y),x_grid,y_grid)
            print("Datapoints after merge:",len(X))
    else:
        #generate simple test data
        #sine curve with some added faulty data
        X = np.arange(-3., 4., 7/40)
        y = [[math.sin(x), 0] for x in X]
        X = np.append(X, [1.,1.,1.,1.,1.])
        y.extend([[1.,0.],[2.,0.],[0.5,0.],[0.,0.], [1.7,0.]])
        
    if (len(X[0]) >= 2):
        dim=2
    X_test = gen_x_test(X, dim, sampling_density)
    y = np.array([[i[0], 0] for i in y])
    model = build_model(neurons, dim, regu, dropout, lr)


    #setup plots
    if(dim==1):
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        pyplot.xlim(np.min(X)-0.8*np.max(X), np.max(X)+0.8*np.max(X))
        pyplot.ylim(np.min([i[0] for i in y])-0.4*np.max([i[0] for i in y]), np.max(y)+0.4*np.max([i[0] for i in y]))
    
        pyplot.autoscale(False)
        pyplot.xlabel('choke')  
        pyplot.ylabel("oil")
        pyplot.show()
        
    f = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
    
    #forward pass function, needed for dropout eval
#    f = K.function([model.layers[0].input, K.learning_phase()],
#                          [model.layers[-1].output])
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
#        if (r%50==0):
#            print ("Run",r)
#            b_func = K.function([model.layers[0].input, K.learning_phase()],
#                                  [model.layers[-2].bias])
#            w_func = K.function([model.layers[0].input, K.learning_phase()],
#                                  [model.layers[-2].kernel])
#            out_func = K.function([model.layers[0].input, K.learning_phase()],
#                                  [model.layers[-1].output])
#            Q=b_func(([[X[3]]],1))
#            W=w_func(([[X[3]]],1))
#            R=out_func(([[X[3]]],1))
#            print("Bias",Q)
#            print("Weights",W)
#            print("Outout",R)



#        #gather results from forward pass
#        results = np.column_stack(f((X_test,1.))[0])
#        if (np.isnan(results[0][0])):
#            print("NAN")
#            return
#
#        res_mean = [results[0]]
#        res_var = [np.exp(results[1])]
#        for i in range(1,n_iter):
#            a = np.column_stack(f((X_test,1.))[0])
#            res_mean.append(a[0])
#            res_var.append(np.exp(a[1]))
        
        #calculate uncertainty
#        pred_mean = np.mean(res_mean, axis=0)
#        pred_sq_mean = np.mean(np.square(res_mean), axis=0)
#        var_mean = np.mean(res_var, axis=0)
#        std = np.sqrt(pred_sq_mean-np.square(pred_mean)+var_mean)
#        
#        var1 = pred_sq_mean-np.square(pred_mean)
        
        #this is the current network's prediction with dropout switched off
        prediction = [x[0] for x in model.predict(X_test)]
        pred_mean, std = sample_mean_std(model, X_test, n_iter, f)
        
        #plot results from current run
        if dim==1:
            if r==0:
                line1 = ax.plot(X, [i[0] for i in y], linestyle='None', marker = '.',markersize=10)
                line2 = ax.plot(X_test,prediction,color='green',linestyle='dashed', linewidth=1)
                for i in range(2):
                    (ax.fill_between([x[0] for x in X_test], pred_mean+std*(i+1), pred_mean-std*(i+1), alpha=0.2, facecolor='#089FFF', linewidth=2))
                line3 = ax.plot(X_test, pred_mean, color='#089FFF', linewidth=1)
#                line3 = ax.plot(X_test, pred_mean, color='#000000', linewidth=1)
            else:
                line2[0].set_ydata(prediction)
                line3[0].set_ydata(pred_mean)
                pyplot.draw()
                ax.collections.clear()
                for i in range(2):
                    (ax.fill_between([x[0] for x in X_test], pred_mean+std*(i+1), pred_mean-std*(i+1), alpha=0.2, facecolor='#089FFF', linewidth=2))
#                    (ax.fill_between([x[0] for x in X_test], pred_mean+var1*(i+1), pred_mean-var1*(i+1), alpha=0.5, facecolor='#0F0F0F', linewidth=2))
#                    (ax.fill_between([x[0] for x in X_test], pred_mean+(var1+var_mean)*(i+1), pred_mean-(var1+var_mean)*(i+1), alpha=0.5, facecolor='#089FFF', linewidth=2))
            pyplot.pause(0.001)
        else:
            if r==0:
                triang, ax = plotter.first_plot_3d([x[0] for x in X], [x[1] for x in X], [x[0] for x in y],[x[0] for x in X_test], [x[1] for x in X_test], pred_mean, well)
            else:
                plotter.update_3d([x[0] for x in X], [x[1] for x in X], [x[0] for x in y], pred_mean, triang, ax)
    
    if (save_weights or save_variance):
        if not scaler:
            model_2 = model
        else:
#            print("Weights before inverse:")
#            for i in range(0,7,3):
#                print(model.layers[i].get_weights())
            model_2 = inverse_scale(model, dim, neurons, dropout, rs, lr)
    
    if (save_weights):
#        print("Weights after inverse:")
#        for i in range(0,7,3):
#            print(model_2.layers[i].get_weights())
        save_variables(well,goal,model_2,case)
    
    if(save_variance):
        if not (scaler == None):
            X_points,y_points,_ = cl.BO_load(well, separator, case=case, scaler=None, goal=goal)        
        X_sample = np.array([[i] for i in range(101)])
        X_save = np.array([i for i in range(101)])
        
        f_scaled = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
#        f = K.function([model_2.layers[0].input, K.learning_phase()], [model_2.layers[-1].output])
        X_sample_scaled = rs.transform(X_sample)
        
#        pred_mean, std = sample_mean_std(model_2, X_sample, n_iter, f)
        pred_mean_scaled, std_scaled = sample_mean_std(model, X_sample_scaled, n_iter, f_scaled)
        std_unscaled = np.array([x[0] for x in rs.inverse_transform(std_scaled.reshape(-1,1))])
        pred_mean_unscaled = np.array([x[0] for x in rs.inverse_transform(pred_mean_scaled.reshape(-1,1))])
#        print(std_unscaled)
#        print(pred_mean_unscaled)
        
        prediction = [x[0] for x in model_2.predict(X_sample)]
        plot_once(X_sample, prediction, pred_mean_unscaled, std, y_points, X_points, extra_points = std_unscaled)
        
#        save_variance_func(X_save, std_unscaled, pred_mean_unscaled, case, well, goal)
        
        
def plot_once(X, prediction, pred_mean, std, y_points, X_points, extra_points=None):
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
#    pyplot.xlim(np.min(X)-0.2*np.max(X), np.max(X)+0.2*np.max(X))
#    pyplot.ylim(np.min([i[0] for i in y_points])-0.4*np.max([i[0] for i in y_points]), np.max(y_points)+0.4*np.max([i[0] for i in y_points]))

#    pyplot.autoscale(False)
    pyplot.xlabel('choke')
    pyplot.ylabel("LOLOLOL")
    pyplot.show()
#    line1 = ax.plot(X_points, [i[0] for i in y_points], linestyle='None', marker = '.',markersize=10)
#    line2 = ax.plot(X,prediction,color='green',linestyle='dashed', linewidth=1)
    for i in range(2):
        (ax.fill_between([x[0] for x in X], pred_mean+std*(i+1), pred_mean-std*(i+1), alpha=0.2, facecolor='#089FFF', linewidth=2))
    line3 = ax.plot(X, pred_mean, color='#089FFF', linewidth=1)
#    if (extra_points is not None):
#        for i in range(2):
#            (ax.fill_between([x[0] for x in X], pred_mean+extra_points*(i+1), pred_mean-extra_points*(i+1), alpha=0.2, facecolor='#089FFF', linewidth=2))
#        line4=ax.plot(X, pred_mean + extra_points,color='green',linestyle="None",marker=".",markersize=5)

def mean_var_to_csv(well, phase="gas", mode="mean", n_iter=200, case=2):
    dims, w, b = tools.load_2(well, phase=phase, case=2, mode=mode)
#    print(w)
#    print(b)
    model = retrieve_model(dims,w,b)
#    for i in range(0,7,3):
#        print(model.layers[i].get_weights())
    X_test = np.array([[x] for x in range(101)])
    prediction = model.predict(X_test)
    pred_mean = prediction[:,0]
#    pred_mean, std = sample_mean_std(model, X_test, n_iter,
#                                     K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output]))
#    plot_once(X_test, 2, pred_mean, std, 1, 1)
    save_variance_func([x for x in range(101)], None, pred_mean, case, well, phase)

    
def retrieve_model(dims, w, b, lr=0.001):
    model_1= Sequential()
    for i in range(1,len(dims)):
        new_w = [np.array(w[i-1]), np.array(b[i-1])]
        model_1.add(Dense(dims[i], input_shape=(dims[i-1],),
                          weights = new_w))
        if (i == len(dims)-1):
            model_1.add(Activation("linear"))
        else:
            model_1.add(Activation("relu"))
            model_1.add(Dropout(0.05))
    model_1.compile(optimizer=optimizers.Adam(lr=lr), loss=sced_loss)
    return model_1
        
def sample_mean_std(model, X, n_iter, f):
    #gather results from forward pass
    results = np.column_stack(f((X,1.))[0])
    if (np.isnan(results[0][0])):
        print("NAN")
        return

    res_mean = [results[0]]
    res_var = [np.exp(results[1])]
    for i in range(1,n_iter):
        a = np.column_stack(f((X,1.))[0])
        res_mean.append(a[0])
        res_var.append(np.exp(a[1]))
        
    pred_mean = np.mean(res_mean, axis=0)
    pred_sq_mean = np.mean(np.square(res_mean), axis=0)
    var_mean = np.mean(res_var, axis=0)
    std = np.sqrt(pred_sq_mean-np.square(pred_mean)+var_mean)
    return pred_mean, std

def gen_x_test(X, dim, n_iter):
    if (dim==1):
        step = (np.max(X)-np.min(X))/n_iter
        X_test = np.array([[i] for i in np.arange(np.min(X)-0.5*np.max(X), 1.5*np.max(X)+step, step)])
#        X_test = np.array([[i] for i in np.arange(0,101)])
    else:
        step_1 = (np.max(X[:,0])-np.min(X[:,0]))/n_iter
        step_2 = (np.max(X[:,1])-np.min(X[:,1]))/n_iter
        X_test = np.array([[i,j] for i in np.arange(np.min(X[:,0])-0.5*(np.max(X[:,0])-np.min(X[:,0])), np.max(X[:,0])+0.5*(np.max(X[:,0])-np.min(X[:,0]))+step_1, step_1)
        for j in np.arange(np.min(X[:,1])-0.5*(np.max(X[:,1])-np.min(X[:,1])), np.max(X[:,1])+0.5*(np.max(X[:,1])-np.min(X[:,1]))+step_2, step_2)])
    return X_test
            
        
        
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

def save_variance_func(X, var, mean, case, well, phase):
    filename = "variance_case" + str(case) +"_"+phase+".csv"
#    well+'_'+phase+"_std":var, 
    try:
        df = pd.read_csv(filename, sep=';', index_col=0)
#        old = pd.read_csv("variance_case_2.csv",sep=";",index_col=0)
        d = {well+"_"+phase+"_mean": mean, well+"_"+phase+"_var": df[well+"_"+phase+"_std"],well+"_"+phase+"_X":X}
        for k, v in d.items():
            df[k] = v
    except Exception as e:
        print(e)
        df = pd.DataFrame(data=d)
        print(df.columns)
    with open(filename, 'w') as f:
        df.to_csv(f,sep=";")
    
def inverse_scale(model_1, dim, neurons, dropout, rs, lr):
    model_2= Sequential()
    model_2.add(Dense(neurons, input_shape=(dim,), weights = [model_1.layers[0].get_weights()[0].reshape(dim,neurons),
                      rs.inverse_transform(model_1.layers[0].get_weights()[1].reshape(-1,1)).reshape(neurons,)]))
    model_2.add(Activation("relu"))
    model_2.add(Dropout(dropout))
    model_2.add(Dense(neurons, weights = [model_1.layers[3].get_weights()[0].reshape(neurons,neurons),
                      rs.inverse_transform(model_1.layers[3].get_weights()[1].reshape(-1,1)).reshape(neurons,)]))
    model_2.add(Activation("relu"))
    model_2.add(Dropout(dropout))
    model_2.add(Dense(2,  weights = [model_1.layers[-2].get_weights()[0],rs.inverse_transform(model_1.layers[-2].get_weights()[1].reshape(-1,1)).reshape(2,)]))
    model_2.add(Activation("linear"))
    model_2.compile(optimizer=optimizers.adam(lr=lr), loss = sced_loss)
    return model_2

def save_variables(well, goal, model, case, hp = 1, is_3d = False, mode="mean"):
    weights = model.get_weights()
    tools.save_variables(well, hp=hp, goal=goal, is_3d = is_3d,
                         neural = weights, case = case, mode = mode)

        
