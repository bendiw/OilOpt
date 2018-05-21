# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:24:34 2018

@author: arntgm
"""

import pandas as pd
from keras.layers import Input, Dense, Activation, LeakyReLU, PReLU, ELU, MaxoutDense, merge, Subtract, Dropout
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from keras import losses, optimizers, backend, regularizers, initializers
import numpy as np
import tools as t
from matplotlib import pyplot
import scipy.stats as ss
# =============================================================================
# This class grossly overfits a PWL NN to data points sampled from a
# distribution with mean and variation from a network trained on well data
# =============================================================================
    

def build_model(neurons, dim, lr, regu=0.0, maxout=False, goal="oil"):
    if maxout:
        a = Input((dim,))
        b = Dense(int(neurons/2))(a)
        c = Dense(int(neurons/2))(a)
        d = MaxoutDense(output_dim=1)(b)
        e = MaxoutDense(output_dim=1)(c)
        f = Subtract()([d,e])
        model_1 = Model(a, f)

    else:        
        #compile model
        model_1= Sequential()
        model_1.add(Dense(neurons, input_shape=(dim,),
                          kernel_initializer=initializers.VarianceScaling(),
                          bias_initializer=initializers.Constant(value=0.1),
                          kernel_regularizer=regularizers.l2(regu),
                          bias_regularizer=regularizers.l2(regu)))
        model_1.add(Activation("relu"))

#        model_1.add(Dense(neurons,
#                          kernel_initializer=initializers.VarianceScaling(),
#                          bias_initializer=initializers.Constant(value=0.1),
#                          kernel_regularizer=regularizers.l2(regu),
#                          bias_regularizer=regularizers.l2(regu)))
#        model_1.add(Activation("relu"))
    #    model_1.add(Dense(neurons,
    #                      kernel_initializer=initializers.VarianceScaling(),
    #                      bias_initializer=initializers.Constant(value=0.1),
    #                      kernel_regularizer=regularizers.l2(regu),
    #                      bias_regularizer=regularizers.l2(regu)))
    #    model_1.add(Activation("relu"))
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
def train_scen(well, goal='gas', neurons=15, dim=1, case=2, lr=0.005,
        epochs=1000, save=False, plot=False, num_std=4, regu=0.0, x_=None, y_=None,
        weight=0.4, iteration=0, points=None, train=False, maxout=False,
        gas_factor = 1000.0, save_sos=False, num_scen=1, scen_start=0, name=""):
    filename = "variance_case"+str(case)+"_"+goal+".csv"
    df = pd.read_csv(filename, sep=';', index_col=0)
    batch_size=7
    for w in well:
        mean_orig = df[str(w)+"_"+goal+"_mean"]
        std_orig = df[str(w)+"_"+goal+"_var"]
        if(points):
            assert(100%points==0)
            factor = int(100/points)
            mean = np.array([mean_orig[i*factor] for i in range(points+1)])
            std = np.array([std_orig[i*factor] for i in range(points+1)])
        X = np.array([[i*factor] for i in range(len(mean))])
        y = np.zeros(len(X))
#        m = np.zeros(len(X))
        if (goal=="gas" and train):
            mean=mean/gas_factor
            std=std/gas_factor
        if(x_ is None):
            index= 0
        else:
            if(x_[w] % factor == 0):
                #point is already in list, so we add another one between two points to ensure same length in scenarios
                index = int(x_[w]/factor)
                insert_index = index
                y = np.append(y,0)
                if (x_[w]<100):
                    X = np.insert(X, index+1, [x_[w]+0.5*factor], axis=0)
                    y[index] = y_[w]
                    insert_index = index + 1
                else:
                    X = np.insert(X, index, [x_[w]-0.5*factor], axis=0)
                    y[index+1] = y_[w]
                interpol_mean = (1-(X[insert_index]-np.floor(X[insert_index]))) * mean_orig[np.floor(X[insert_index])] + (X[insert_index]-np.floor(X[insert_index])) * mean_orig[np.ceil(X[insert_index])]
                interpol_std = (1-(X[insert_index]-np.floor(X[insert_index]))) * std_orig[np.floor(X[insert_index])] + (X[insert_index]-np.floor(X[insert_index])) * std_orig[np.ceil(X[insert_index])]
            else:
                X = np.append(X, [[x_[w]]], axis=0)
                X = np.sort(X,axis=0)
                index = np.where(X==x_[w])[0][0]
                insert_index = index
                y=np.insert(y, index, y_[w])
                interpol_mean = (1-(x_[w]-np.floor(x_[w]))) * mean_orig[np.floor(x_[w])] + (x_[w]-np.floor(x_[w])) * mean_orig[np.ceil(x_[w])]
#                print(interpol_mean)
                interpol_std = (1-(x_[w]-np.floor(x_[w]))) * std_orig[np.floor(x_[w])] + (x_[w]-np.floor(x_[w])) * std_orig[np.ceil(x_[w])]
            mean = np.insert(mean, insert_index, interpol_mean)
            std = np.insert(std, insert_index, interpol_std)
        y[0] = 0

        for scen in range(scen_start, scen_start+num_scen):
#            for i in range(len(X)):
#                m[i] = mean[i]
            for i in range(index+1,len(X)):
    #                y[i] = (1-weight)*y[i-1] + weight*ss.truncnorm.rvs(-num_std, num_std, scale=std[i], loc=mean[i], size=(1))
                y[i] = max(0, (1-weight)*(mean[i]+std[i]*((y[i-1]-mean[i-1])/std[i-1])) + weight*ss.truncnorm.rvs(-num_std, num_std, scale=std[i], loc=mean[i], size=(1)))
            for i in range(index-1,0,-1):
    #                y[i] = max((1-weight)*y[i+1] + weight*ss.truncnorm.rvs(-num_std, num_std, scale=std[i], loc=mean[i], size=(1)),0)
                y[i] = max(0, (1-weight)*(mean[i]+std[i]*((y[i+1]-mean[i+1])/std[i+1])) + weight*ss.truncnorm.rvs(-num_std, num_std, scale=std[i], loc=mean[i], size=(1)))
            if(train):
                early_stopping = EarlyStopping(monitor='loss', patience=10000, verbose=0, mode='auto')
                model = build_model(neurons, dim, lr, regu=regu)
#                for i in range(100):
#                model.fit(X,y,batch_size=batch_size,epochs=int(epochs),verbose=0)
                model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[early_stopping])
                prediction = [x[0] for x in model.predict(X)]
#                ax = plot_all(X, y, prediction, mean, std, m, goal, weight, points, x_, y_, w, train, ax)
                    

            else:
                prediction = None
            if plot or save:
                if goal=="gas" and train:
                    model = t.add_layer(model,neurons,"mse", factor=gas_factor)
                    prediction = [x[0] for x in model.predict(X)]
#                    m = m*gas_factor
                    y = y*gas_factor
                    std = std*gas_factor
                    mean = mean*gas_factor
                plot_all(X, y, prediction, mean, std, goal, weight, points, x_[w], y_[w], w, train)
                if save:
                    filepath = "scenarios\\nn\\points\\"+w+"_"+str(scen)+".png"
                    pyplot.savefig(filepath, bbox_inches="tight")
                    t.save_variables(w+"_"+str(scen), goal=goal, case=2,neural=model.get_weights(), mode="scen", folder="scenarios\\nn\\points\\")
                if plot:
                    pyplot.show()
            if (save_sos):
                save_sos2(X,y,goal,w, scen, folder="scenarios\\nn\\points\\", name=name)
#        m = np.zeros(len(X))
#

def plot_all(X, y, prediction, mean, std, goal, weight, points, x_, y_, w, train, prev=None):
    if prev is None:
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        line1 = ax.plot(X, y,color="green",linestyle="dashed", linewidth=1)
        if(x_ is not None):
            line1 = ax.plot([x_], [y_],color="red",linestyle="None", marker=".", markersize=7)
    
        line3 = ax.plot(X, mean, color="black", linewidth=0.7)
        if(train):
            line2 = ax.plot(X, prediction, color='red',linestyle='dashed', linewidth=1)
        else:
            line2=None
        pyplot.title(w+", weight="+ str(round(weight, 1))+", points="+str(points))
        pyplot.xlabel('choke')
        pyplot.ylabel(goal)
        pyplot.fill_between([x[0] for x in X], mean-std, mean+std,
                           alpha=0.2, facecolor='#089FFF', linewidth=1)
        pyplot.fill_between([x[0] for x in X], mean-2*std, mean+2*std,
                           alpha=0.2, facecolor='#089FFF', linewidth=1)
    else:
        ax, line1, line2, line3 = prev[0],prev[1],prev[2],prev[3]
        line1[0].set_ydata(y)
        if(x_ is not None):
            line1[0].set_ydata([y_])
        line3[0].set_ydata(m)
        if(train):
            line2[0].set_ydata(prediction)
        ax.collections.clear()
        pyplot.title(w+", weight="+ str(round(weight, 1))+", points="+str(points))
        pyplot.xlabel('choke')
        pyplot.ylabel(goal)
        pyplot.fill_between([x[0] for x in X], mean-std, mean+std,
                           alpha=0.2, facecolor='#089FFF', linewidth=1)
        pyplot.fill_between([x[0] for x in X], mean-2*std, mean+2*std,
                           alpha=0.2, facecolor='#089FFF', linewidth=1)
    pyplot.pause(0.001)
    return [ax, line1, line2, line3]


def train_all_scen(neurons=15,lr=0.005,epochs=1000,save=True,plot=False, case=2, num_std=4):
    for w in t.wellnames_2:
        for p in ["oil","gas"]:
            train_scen(w, goal=p, neurons=neurons, lr=lr, epochs=epochs, save=save, plot=plot, case=case, num_std=num_std)

def save_sos2(X,y,phase, well, scen, folder, name=""):
    filename = folder + "sos2_" +phase+"_"+name+".csv"
#    well+'_'+phase+"_std":var, 
#    x = [z[0] for z in x]
    d = {well+"_"+phase+"_"+str(scen): y}
    try:
        df = pd.read_csv(filename, sep=';', index_col=0)
#        old = pd.read_csv("variance_case_2.csv",sep=";",index_col=0)
        if not (well+"_choke" in df.keys()):
            d[well+"_choke"] = X
        for k, v in d.items():
            df[k] = v
    except Exception as e:
        print("Exception:", e)
        d[well+"_choke"] = np.array([x[0] for x in X])
        df = pd.DataFrame(data=d)
#        print(df.columns)
        
    with open(filename, 'w') as f:
        df.to_csv(f,sep=";")
        
        
def sos2_to_nn(well,epochs, phase="gas", num_scen=10, start_scen=0, scens=[], neurons=20, lr=0.001):
    df = t.get_sos2_scenarios(phase, start_scen+num_scen)
    X = np.array([[i*10] for i in range(11)])
    for scen in range(start_scen, start_scen+num_scen):
        train(well, X, df[scen][well], goal=phase, neurons=neurons, lr=lr,
              epochs=epochs, save=True, plot=True, scen=scen)
        
    
    
def train(well, X, y, goal='gas', neurons=15, dim=1, case=2, lr=0.005,
               epochs=1000, save=False, plot=False, regu=0.0, scen = 0,
               gas_factor = 1000.0, num_scen=1, scen_start=0):
    
    batch_size=7
    if (goal=="gas"):
        y = y/gas_factor

    early_stopping = EarlyStopping(monitor='loss', patience=20000, verbose=1, mode='auto')
    model = build_model(neurons, dim, lr, regu=regu)
#                for i in range(100):
#                model.fit(X,y,batch_size=batch_size,epochs=int(epochs),verbose=0)
    print("Fitting to data:",y)
    model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[early_stopping])
    prediction = [x[0] for x in model.predict(X)]
#                ax = plot_all(X, y, prediction, mean, std, m, goal, weight, points, x_, y_, w, train, ax)
    
    if plot or save:
        if goal=="gas" and train:
            model = t.add_layer(model,neurons,"mse", factor=gas_factor)
            prediction = [x[0] for x in model.predict(X)]
            y = y*gas_factor
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        line1 = ax.plot(X, y,color="green",linestyle="None", marker=".", markersize=10)    
        line2 = ax.plot(X, prediction, color="blue", linestyle="dashed", linewidth=1)
        if save:
            filepath = "scenarios\\nn\\points\\"+well+"_"+str(scen)+".png"
            pyplot.savefig(filepath, bbox_inches="tight")
            t.save_variables(well+"_"+str(scen), goal=goal, case=2,neural=model.get_weights(), mode="scen", folder="scenarios\\nn\\points\\")
        if plot:
            pyplot.show()


            
    
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