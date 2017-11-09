import pandas as pd
import numpy as np
import math

def load(path, well=None):
    df = pd.read_csv(path, sep=",")
    if(well):
        df = df.loc[df['well'] == well]
    return df

def gen_test_case(cases=30):
    X, Y = [], []
    data = []
    for i in range(cases):
        x = r.uniform(-10,10)
        X.append(x)
    X.sort()
    for x in X[:cases-5]:
        y = x**2 + r.uniform(-abs(x),abs(x))
        Y.append(y)
        data.append([[x],[y]])
    for x in X[cases-5:]:
        y=x/2
        Y.append(y)
        data.append([[x],[y]])
    return data


##############
##              Generates targets from a dataframe                      ##
## If intervals input is given, the dataset will be processed to reduce ##
## amount of data points in close proximity to each other. The way this ##
## is done is specified by the mode parameter.                          ##
##############
def gen_targets(df, well, intervals=None, allow_nan=False, normalize = False, factor=0, nan_ratio=0.5):
    df = df.loc[df['well']==well]
    ret = {}
    add_Z = False
    if not (df['choke'].isnull().sum()/df.shape[0] > nan_ratio):
        add_Z = True
        intervals = None
        Z = []
    else:
        Z = None
    if(not allow_nan):
        df = df[pd.notnull(df['gaslift_rate'])]
        df = df[pd.notnull(df['oil'])]
        if(add_Z):
            df = df[pd.notnull(df['choke'])]
    if(intervals):
        X = []
        y = []
        min_g = df['gaslift_rate'].min()
        max_g = df['gaslift_rate'].max()
##        if(add_Z):
##            min_c = df['choke'].min()
##            max_c = df['choke'].max()
##            c_diff = max_c-min_c
##            c_step = c_diff/intervals
##            c_vals = np.arange(min_c, max_c, c_step)
        g_diff = max_g-min_g
        step = g_diff/intervals
        print(step)
        vals = np.arange(min_g, max_g, step)
        for i in range(len(vals)):
            val = df.loc[df['gaslift_rate']>=vals[i]]
            val = val.loc[val['gaslift_rate']<=vals[i]+step]
            maxtime = val['time_ms_begin'].max()
            mintime = val['time_ms_begin'].min()
            div = maxtime-mintime
            if(val.shape[0] > 1):
                factors = val['time_ms_begin'].apply(lambda x: math.exp(-factor*((maxtime-x)/div)))
                oil = val['oil'].multiply(factors).sum() / factors.sum()
                glift = val['gaslift_rate'].multiply(factors).sum() / factors.sum()
####                if(add_Z):
##                    choke = val['choke'].multiply(factors).sum() / factors.sum()
            elif(val.shape[0]==1):
                glift = val['gaslift_rate'].values[0]
                oil = val['oil'].values[0]
##                if(add_Z):
##                    choke = val['choke'].values[0]
            if(not val.empty):
                X.append(glift)
                y.append(oil)
##                if(add_Z):
##                    Z.append(choke)
    else:
        if add_Z:
            Z = df['choke']
        X = df['gaslift_rate']
        y = df['oil']
        
##    print(X)
##    print(y)
    if(normalize):
        X = normalize_data(X)
        y = normalize_data(y)

        if add_Z:
            Z = normalize_data(Z)
    ret['gaslift'] = np.array(X)
    ret['output'] = np.array(y)
    if(add_Z):
        ret['choke'] = np.array(Z)
    return ret

def normalize_data(data):
    X_mean = np.mean(data)
    X_std = np.std(data)
    X = [(x-X_mean)/X_std for x in data]
    return X

def conv_to_batch_multi(X, Y, Z):
    batch = []
    for i in range(len(X)):
        tup = [ [X[i], Y[i]], Z[i] ]
        batch.append(tup)
    return batch

def conv_to_batch(data):
    batch= []
####    print(data[0])
    for i in range(len(data[0])):
        tup = [[data[0][i]], [data[1][i]]]
##        tup.append([data[0][0][i]])
##        tup.append([data[0][1][i]])
        batch.append(tup)
    return batch
