import numpy as np
from pyearth import Earth
import pandas as pd
from matplotlib import pyplot
import math
import caseloader as cl


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

    X = np.array(X)
    y = np.array(y)
    print("\nwell: ", well)
    print(min_s, max_s)
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
