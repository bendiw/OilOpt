import numpy as np
from pyearth import Earth
import pandas as pd
from matplotlib import pyplot
import math


base_dir = "C:\\Users\\Bendik\\Documents\\GitHub\\OilOpt\\"
data_file = "welltests.csv"

df = pd.read_csv(base_dir+data_file, sep=",")
df = df[pd.notnull(df['gaslift_rate'])]
df = df[pd.notnull(df['oil'])]

intervals = 100
##df.columns = [c.
##print(df.columns[1])
##print(df.dtypes[1])
##print(df.loc['choke'])
well='A3'
model = Earth(allow_missing = True, enable_pruning=False, max_terms=10, penalty=0.1,
              minspan=3)

for well in df.well.unique():
    df_w = df.loc[df['well'] == well]

    min_s = df_w['time_ms_begin'].min()
    max_s = df_w['time_ms_begin'].max()
    t_diff = max_s-min_s
##    df_w = df_w.loc[df_w['time_ms_begin']>= max_s-(t_diff/2)]
    min_g = df_w['gaslift_rate'].min()
    max_g = df_w['gaslift_rate'].max()
    g_diff = max_g-min_g
    step = g_diff/intervals
    X = []
    y = []
    vals = np.arange(min_g, max_g, step)
    #print(df_w.loc[df_w['gaslift_rate']>0])
    for i in vals:
        val = df_w.loc[df_w['gaslift_rate']>=i-step]
        val = val.loc[val['gaslift_rate']<=i+step]
        #print(val.shape[0])
        #print(val.empty)
##        print(val[['gaslift_rate', 'oil']])
        if(val.shape[0] >= 1):
            glift, oil = val.ix[val['time_ms_begin'].idxmax()][['gaslift_rate', 'oil']]
        elif(not val.empty):
            glift, oil = val[['gaslift_rate','oil']]
        if(not val.empty):
            X.append(glift)
            y.append(oil)
##    X = df_w['gaslift_rate']
##    y = df_w['oil']
    X = np.array(X)
    y = np.array(y)
##    print(X)
    print("\nwell: ", well)
    print(min_s, max_s)
##    print(min_g, max_g)
####    X = X.dropna()
##    y = y.dropna()
    #print(X, y)
    model.fit(X, y)

##    print(model.trace())
##    print(model.summary())
    print((max_s-min_s)/(60*60*24*365*1000))

    y_hat = model.predict(X)
    pyplot.figure()
    pyplot.plot(X,y,'r.')
    pyplot.plot(X,y_hat,'b.')
    pyplot.xlabel('gaslift')
    pyplot.ylabel('oil')
    pyplot.title(well)
    pyplot.show()
