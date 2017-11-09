import numpy as np
import pyearth
from pyearth import Earth
import pandas as pd
from matplotlib import pyplot
import math
import caseloader as cl
import plotter
import random as r

data_file = "welltests.csv"

class Mars:

    def __init__(self, miss = True, prune=False, max_terms=10, penalty=0.1, minspan=1):
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
        d_dict = cl.gen_targets(self.df, well, normalize=True, allow_nan = False, nan_ratio=0.9, intervals=100, factor=0)
        if('choke' in d_dict.keys()):
            data = cl.conv_to_batch_multi(d_dict['gaslift'], d_dict['choke'], d_dict['output'])
        else:
            data = cl.conv_to_batch([d_dict['gaslift'], d_dict['output']])
        data.sort()
####        data_sorted = cl.conv_to_batch([X,y]).sort()
        X = [d[0] for d in data]
        y = [d[1] for d in data]
        print(X[0])
        self.model.fit(X, y)
        print(self.model.summary())
        print("R2 score: ", self.model.score(X, y), "\n")
        y_hat = self.model.predict(X)
        X = [x[0] for x in X]
        if('choke' in d_dict.keys()):
            #self.plot_fig(X, y, y_hat, well, brk=self.get_multi_breakpoints())
            self.test3d(d_dict, 50)
##            plotter.mesh(d_dict['gaslift'], d_dict['choke'], d_dict['output'])

        else:
            self.plot_fig(X, y, y_hat, well, brk=self.get_breakpoints())

    def get_breakpoints(self):
        brk = []
        for bf in self.model.basis_:
##            print(type(bf))
            if type(bf) is pyearth._basis.HingeBasisFunction:
                if(not bf.is_pruned()):                    
                    brk.append([bf.get_knot(), self.model.predict([bf.get_knot(),])[0]])
        return brk

    def get_multi_breakpoints(self):
        brk = []
        for bf in self.model.basis_:
##            print(type(bf))
            if type(bf) is pyearth._basis.HingeBasisFunction:
                if(not bf.is_pruned()):
                    print(bf.get_knot())
                    #brk.append([bf.get_knot(), self.model.predict([bf.get_knot(),])[0]])
        return brk

    def test3d(self, d_dict, inters):
        t_z = []
        t_x = []
        t_y = []
        x, y, z = d_dict['gaslift'], d_dict['choke'], d_dict['output']
        x_v = np.arange(np.nanmin(x), np.nanmax(x), (np.nanmax(x)-np.nanmin(x))/inters)
        y_v = np.arange(np.nanmin(y), np.nanmax(y), (np.nanmax(y)-np.nanmin(y))/inters)
        for i in x_v:
            for j in y_v:
                t_x.append(i)
                t_y.append(j)
                t_z.append(self.model.predict([[i, j]]))
        plotter.plot3d(t_x, t_y, [m[0] for m in t_z])



def run_all():
    m = Mars()
    wells = m.df.well.unique()
    for well in wells:
        m.run_well(well)

def run(well):
    m = Mars()
    m.run_well(well)


