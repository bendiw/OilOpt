import numpy as np
from pyearth import Earth
import pandas as pd
from matplotlib import pyplot
import math
import caseloader as cl
import random as r

data_file = "welltests.csv"

class Mars:

    def __init__(self, miss = True, prune=False, max_terms=6, penalty=0.1, minspan=3):
        self.model = Earth(allow_missing=miss, enable_pruning=prune, max_terms=max_terms, penalty=penalty,
                  minspan=minspan)
        dframe = pd.read_csv(data_file, sep=",")
        dframe = dframe[pd.notnull(dframe['gaslift_rate'])]
        dframe = dframe[pd.notnull(dframe['oil'])]
        self.df = dframe
        
    def plot_fig(self, X, y, y_hat, title="Model fit"):
        pyplot.figure()
        pyplot.plot(X,y,'r.')
        pyplot.plot(X,y_hat,'b.')
        pyplot.xlabel('gaslift')
        pyplot.ylabel('oil')
        pyplot.title(title)
        pyplot.show()

    def run_well(self, well):
##        self.model = Earth(allow_missing=True, enable_pruning=False, max_terms=6, penalty=0.1,
##                  minspan=6)
        df_w = self.df.loc[self.df['well'] == well]
        X,y = cl.gen_targets(self.df, well, normalize=True, intervals=100)
        X = np.array(X)
        y = np.array(y)
        self.model.fit(X, y)
    ##    print(model.trace())
        print(self.model.summary())
        y_hat = self.model.predict(X)
        self.plot_fig(X, y, y_hat, well)

def run_all():
    m = Mars()
    wells = m.df.well.unique()
    for well in wells:
        m.run_well(well)

def run(well):
    m = Mars()
    m.run_well(well)


