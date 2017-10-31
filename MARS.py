import numpy as np
from pyearth import Earth
import pandas as pd
from matplotlib import pyplot
import math
import caseloader as cl
import random as r

def plot_fig(X, y, y_hat):
    pyplot.figure()
    pyplot.plot(X,y,'r.')
    pyplot.plot(X,y_hat,'b.')
    pyplot.xlabel('gaslift')
    pyplot.ylabel('oil')
    pyplot.title(well)
    pyplot.show()


base_dir = "C:\\Users\\Bendik\\Documents\\GitHub\\OilOpt\\"
data_file = "welltests.csv"

df = pd.read_csv(base_dir+data_file, sep=",")
df = df[pd.notnull(df['gaslift_rate'])]
df = df[pd.notnull(df['oil'])]

well='A5'
model = Earth(allow_missing = True, enable_pruning=False, max_terms=6, penalty=0.1,
              minspan=3)
wells = df.well.unique()


X, Y = [], []
data = []
cases = 20
for i in range(cases):
    x = r.uniform(-10,10)
    X.append(x)
X.sort()
for x in X:
    y = x**2 + r.uniform(-abs(x),abs(x))
    Y.append(y)
    data.append([[x],[y]])
model.fit(np.array(X), np.array(y))
y_hat = model.predict(X)
plot_figure(X, y, y_hat)

for well in []:
    df_w = df.loc[df['well'] == well]
    df = cl.load(base_dir+data_file)
    X,y = cl.gen_targets(df, well, normalize=True, intervals=100)
    X = np.array(X)
    y = np.array(y)
    model.fit(X, y)

##    print(model.trace())
    print(model.summary())

    y_hat = model.predict(X)
    plot_figure(X, y, y_hat)
