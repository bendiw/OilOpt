import numpy as np
from pyearth import Earth
import pandas as pd
from matplotlib import pyplot
import math
import caseloader as cl
import random as r

data_file = "welltests.csv"

def plot_fig(X, y, y_hat, title="Model fit"):
    pyplot.figure()
    pyplot.plot(X,y,'r.')
    pyplot.plot(X,y_hat,'b.')
    pyplot.xlabel('gaslift')
    pyplot.ylabel('oil')
    pyplot.title(title)
    pyplot.show()

def run_all_wells():
    model = Earth(allow_missing = True, enable_pruning=False, max_terms=6, penalty=0.1,
              minspan=3)
    df = pd.read_csv(data_file, sep=",")
    df = df[pd.notnull(df['gaslift_rate'])]
    df = df[pd.notnull(df['oil'])]
    wells = df.well.unique()
    for well in wells:
        df_w = df.loc[df['well'] == well]
        df = cl.load(base_dir+data_file)
        X,y = cl.gen_targets(df, well, normalize=True, intervals=100)
        X = np.array(X)
        y = np.array(y)
        model.fit(X, y)

    ##    print(model.trace())
        print(model.summary())

        y_hat = model.predict(X)
        plot_fig(X, y, y_hat, well)


run_all_wells()

df = pd.read_csv(base_dir+data_file, sep=",")
df = df[pd.notnull(df['gaslift_rate'])]
df = df[pd.notnull(df['oil'])]

well='A5'



model.fit(np.array(X), np.array(y))
y_hat = model.predict(X)
plot_figure(X, y, y_hat)


