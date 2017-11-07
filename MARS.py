import numpy as np
import pyearth
from pyearth import Earth
import pandas as pd
from matplotlib import pyplot
import math
import caseloader as cl
import random as r

data_file = "welltests.csv"

class Mars:

    def __init__(self, miss = True, prune=False, max_terms=10, penalty=0.1, minspan=3):
        self.model = Earth(allow_missing=miss, enable_pruning=prune, max_terms=max_terms, penalty=penalty,
                  minspan=minspan)
        dframe = pd.read_csv(data_file, sep=",")
        dframe = dframe[pd.notnull(dframe['gaslift_rate'])]
        dframe = dframe[pd.notnull(dframe['oil'])]
        self.df = dframe
        
    def plot_fig(self, X, y, y_hat, title="Model fit", brk=None):
        pyplot.figure()
##        X = [d[0] for d in data]
####        y = [d[1] for d in data]
        pyplot.plot(X,y,'b.')
        pyplot.plot(X,y_hat,'r')
        pyplot.xlabel('gaslift')
        pyplot.ylabel('oil')
        pyplot.title(title)
        if(brk):
            for pair in brk:
                pyplot.plot(pair[0], pair[1], 'k*')
        pyplot.show()

    def run_well(self, well):
        df_w = self.df.loc[self.df['well'] == well]
##        X,y = cl.gen_targets(self.df, well, normalize=True)
        data = cl.conv_to_batch([cl.gen_targets(self.df, well, normalize=True, intervals=100, factor=0)])
        data.sort()
####        data_sorted = cl.conv_to_batch([X,y]).sort()
        X = [d[0] for d in data]
        y = [d[1] for d in data]
        self.model.fit(X, y)
    ##    print(model.trace())
        print(self.model.summary())
        print("R2 score: ", self.model.score(X, y), "\n")
##        print(dir(self.model.basis_[2]))
##        print((self.model.basis_[0].get_knot()))
        y_hat = self.model.predict(X)
        #print(y_hat)
        X = [x[0] for x in X]
        #print("\n", X, "\n\n")
##        print(self.model.predict([98988,]))
        self.plot_fig(X, y, y_hat, well, brk=self.get_breakpoints())

    def get_breakpoints(self):
        brk = []
        for bf in self.model.basis_:
##            print(type(bf))
            if type(bf) is pyearth._basis.HingeBasisFunction:
                if(not bf.is_pruned()):                    
                    brk.append([bf.get_knot(), self.model.predict([bf.get_knot(),])[0]])
        return brk


def run_all():
    m = Mars()
    wells = m.df.well.unique()
    for well in wells:
        m.run_well(well)

def run(well):
    m = Mars()
    m.run_well(well)


