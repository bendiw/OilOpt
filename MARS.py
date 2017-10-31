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

##df.columns = [c.
##print(df.columns[1])
##print(df.dtypes[1])
##print(df.loc['choke'])
well='A3'
model = Earth(allow_missing = True, enable_pruning=False)

for well in df.well.unique():
    df_w = df.loc[df['well'] == well]
    X = df_w['gaslift_rate']
    y = df_w['oil']
    min_s = df_w['time_ms_begin'].min()
    max_s = df_w['time_ms_begin'].max()
    ####X = X.dropna()
    ####y = y.dropna()
    print(X, y)
    model.fit(X, y)

##    print(model.trace())
    print(model.summary())
    print((max_s-min_s)/(60*60*24*365*1000))

    y_hat = model.predict(X)
    pyplot.figure()
    pyplot.plot(X,y,'r.')
    pyplot.plot(X,y_hat,'b.')
    pyplot.xlabel('gaslift')
    pyplot.ylabel('oil')
    pyplot.title(well)
    pyplot.show()
