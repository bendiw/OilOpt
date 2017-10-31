import pandas as pd
import numpy as np

def load(path, well=None):
    df = pd.read_csv(path, sep=",")
    if(well):
        df = df.loc[df['well'] == well]
    return df

def gen_targets(df, well, intervals=None, allow_nan=False):
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
            val = df.loc[df['gaslift_rate']>=i-step]
            val = val.loc[val['gaslift_rate']<=i+step]
            if(val.shape[0] >= 1):
                glift, oil = val.ix[val['time_ms_begin'].idxmax()][['gaslift_rate', 'oil']]
            elif(not val.empty):
                glift, oil = val[['gaslift_rate','oil']]
            if(not val.empty):
                X.append(glift)
                y.append(oil)
    else:
        X = df['gaslift_rate']
        y = df['oil']
    return np.array(X),np.array(y)

def conv_to_batch(data):
    batch= []
    for i in range(len(data[0][0])):
        tup = []
        tup.append([data[0][0][i]])
        tup.append([data[0][1][i]])
        batch.append(tup)
    return batch
