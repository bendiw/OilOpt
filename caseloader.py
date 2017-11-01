import pandas as pd
import numpy as np

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
def gen_targets(df, well, intervals=None, allow_nan=False, normalize = False, mode="new"):
    df = df.loc[df['well']==well]
    if(not allow_nan):
        df = df[pd.notnull(df['gaslift_rate'])]
        df = df[pd.notnull(df['oil'])]
    if(intervals):
        X = []
        y = []
        min_g = df['gaslift_rate'].min()
        max_g = df['gaslift_rate'].max()
        g_diff = max_g-min_g
        step = g_diff/intervals
        vals = np.arange(min_g, max_g, step)
        for i in vals:
            val = df.loc[df['gaslift_rate']>=i]
            val = val.loc[val['gaslift_rate']<=i+step]
            if(val.shape[0] > 1):
##                print(val)
                if(mode=="avg"):
                    print(val[['gaslift_rate', 'oil']])
                    df_m = val[['gaslift_rate', 'oil']].mean(axis=0)
                    print(df_m)
                    glift = df_m['gaslift_rate']
                    oil = df_m['oil']
                else:
                    glift, oil = val.ix[val['time_ms_begin'].idxmax()][['gaslift_rate', 'oil']]

            elif(val.shape[0]==1):
                glift = val['gaslift_rate'].values[0]
                oil = val['oil'].values[0]
            if(not val.empty):
                X.append(glift)
                y.append(oil)
    else:
        X = df['gaslift_rate']
        y = df['oil']
##    print(X)
##    print(y)
    if(normalize):
        X, y = normalize_data(X, y)
    return np.array(X),np.array(y)

def normalize_data(X, y):
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    X_std = np.std(X)
    y_std = np.std(y)
    X = [(x-X_mean)/X_std for x in X]
    y = [(y-y_mean)/y_std for y in y]
    return X, y

def conv_to_batch(data):
    batch= []
    for i in range(len(data[0][0])):
        tup = []
        tup.append([data[0][0][i]])
        tup.append([data[0][1][i]])
        batch.append(tup)
    return batch
