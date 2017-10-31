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
    X = df.loc[df['well'] == well, 'gaslift_rate']
    y = df.loc[df['well'] == well, 'oil']
    ####X = X.dropna()
    ####y = y.dropna()
    print(X, y)
    model.fit(X, y)

##    print(model.trace())
    print(model.summary())

    y_hat = model.predict(X)
    pyplot.figure()
    pyplot.plot(X,y,'r.')
    pyplot.plot(X,y_hat,'b.')
    pyplot.xlabel('gaslift')
    pyplot.ylabel('oil')
    pyplot.title(well)
    pyplot.show()
