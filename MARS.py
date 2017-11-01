import numpy as np
from pyearth import Earth
import pandas as pd
from matplotlib import pyplot
import math
import caseloader as cl
import random as r

data_file = "welltests.csv"

class Mars:

    def __init__(self, miss = True, prune=False, max_terms=30, penalty=0.0, minspan=3):
        self.model = Earth(allow_missing=miss, enable_pruning=prune, max_terms=max_terms, penalty=penalty,
                  minspan=minspan)
        dframe = pd.read_csv(data_file, sep=",")
        dframe = dframe[pd.notnull(dframe['gaslift_rate'])]
        dframe = dframe[pd.notnull(dframe['oil'])]
        self.df = dframe
        
    def plot_fig(self, X, y, y_hat, title="Model fit"):
        pyplot.figure()
##        X = [d[0] for d in data]
####        y = [d[1] for d in data]
        pyplot.plot(X,y,'b.')
        pyplot.plot(X,y_hat,'r')
        pyplot.xlabel('gaslift')
        pyplot.ylabel('oil')
        pyplot.title(title)
        pyplot.show()

    def run_well(self, well):
##        self.model = Earth(allow_missing=True, enable_pruning=False, max_terms=6, penalty=0.1,
##                  minspan=6)
        df_w = self.df.loc[self.df['well'] == well]
##        X,y = cl.gen_targets(self.df, well, normalize=True)
        data = cl.conv_to_batch([cl.gen_targets(self.df, well, normalize=False, intervals=100)])
        data.sort()
####        data_sorted = cl.conv_to_batch([X,y]).sort()
        X = [d[0] for d in data]
        y = [d[1] for d in data]
        self.model.fit(X, y)
    ##    print(model.trace())
        print(self.model.summary())
        y_hat = self.model.predict(X)
        print(y_hat)
        X = [x[0] for x in X]
        print("\n", X, "\n\n")
        self.plot_fig(X, y, y_hat, well)


def run_all():
    m = Mars()
    wells = m.df.well.unique()
    for well in wells:
        m.run_well(well)

def run(well):
    m = Mars()
    m.run_well(well)


