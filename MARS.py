import numpy as np
import pyearth
from pyearth import Earth
import pandas as pd
from matplotlib import pyplot
import math
import caseloader as cl
import plotter
import random as r
import tools
from importlib import reload

data_file = "welltests_new.csv"
reload(plotter)
reload(tools)

class Mars:
    
# =============================================================================
#     Initialises a Mars object
# =============================================================================
    def __init__(self, prune, max_terms, penalty, minspan, miss = True):
        self.model = Earth(allow_missing=miss, enable_pruning=prune, max_terms=max_terms, penalty=penalty,
                  minspan=minspan)
        dframe = pd.read_csv(data_file, sep=",")
        dframe = dframe[pd.notnull(dframe['gaslift_rate'])]
        dframe = dframe[pd.notnull(dframe['oil'])]
        
#        z = df_2.loc[df_2["well"]=="B2"]
#        print(z.loc[z["prs_dns"] < 18.5])
        
        self.df = dframe
        
# =============================================================================
#      plots a 2D chart with oil output and gas lift   
# =============================================================================
    def plot_fig(self, X, y, y_hat, title="Model fit", brk=None, goal='oil'):
        pyplot.figure()
##        X = [d[0] for d in data]
####        y = [d[1] for d in data]
        pyplot.plot(X,y,'c.', markersize=8)
        pyplot.plot(X,y_hat,'b')
        pyplot.xlabel('gaslift')
        pyplot.ylabel(goal)
        pyplot.title(title)
        if(brk):
            for pair in brk:
                pyplot.plot(pair[0], pair[1], 'k*')
        pyplot.show()

# =============================================================================
# runs the training of a single well
# =============================================================================
    def run_well(self, well, goal, plot, normalize, hp, allow_nan, nan_ratio, intervals, factor, grid_size, verbose, train_frac=0, val_frac=0):
#        df_w = self.df.loc[self.df['well'] == well]
#        print(df_w.shape)
##        X,y = cl.gen_targets(self.df, well, normalize=True)
        d_dict, means, std = cl.gen_targets(self.df, well,goal=goal, normalize=normalize, allow_nan = allow_nan, nan_ratio=nan_ratio, intervals=intervals, factor=factor, hp=hp)
#        print(d_dict)
        if('choke' in d_dict.keys()):
            data = cl.conv_to_batch_multi(d_dict['gaslift'], d_dict['choke'], d_dict['output'])
        else:
            data = cl.conv_to_batch([d_dict['gaslift'], d_dict['output']])
        if (val_frac > 0.0 or train_frac - val_frac > 0.0):
            train_set, val_set, test_set = self.generate_sets(data, train_frac, val_frac)
        else:
            train_set=data
            train_set.sort()
####        data_sorted = cl.conv_to_batch([X,y]).sort()s
        X = [d[0] for d in train_set]
        y = [d[1] for d in train_set]
        X_val = [d[0] for d in val_set]
        y_val = [d[1] for d in val_set]
        X_test = [d[0] for d in test_set]
        y_test = [d[1] for d in test_set]
        try:
            float(y_test[0])
            y_test = [[d] for d in y_test]
        except TypeError:
            a=0

#        print(X, y)
        self.model.fit(X, y)
        if(verbose >2):
            print("********* WELL", well, "*********")
    #        print(self.model.summary())
            print("R2 score: ", self.model.score(X, y))
            print("******************", "\n\n")
        y_hat = self.model.predict(X)
        X = [x[0] for x in X]
        
#        if('choke' in d_dict.keys()):
##            plotter.mesh(d_dict['gaslift'], d_dict['choke'], d_dict['output'])
#            self.plot_fig(X, y, y_hat, well, brk=self.get_multi_breakpoints())
        if('choke' in d_dict.keys()):
                
            if(plot):
                self.test3d(d_dict, grid_size, well)
            t_z = []
            t_x = []
            t_y = []
            x, y, z = d_dict['gaslift'], d_dict['choke'], d_dict['output']
            x_v = np.arange(np.nanmin(x), np.nanmax(x), (np.nanmax(x)-np.nanmin(x))/grid_size)
            y_v = np.arange(np.nanmin(y), np.nanmax(y), (np.nanmax(y)-np.nanmin(y))/grid_size)
            for i in x_v:
                for j in y_v:
                    t_x.append(i)
                    t_y.append(j)
                    t_z.append(self.model.predict([[i, j]]))
            return True, tools.delaunay(t_x, t_y, [m[0] for m in t_z]), X,y,y_hat,X_test,y_test #[m[0] for m in t_z]
        else:
            if(plot):
                self.plot_fig(X, y, y_hat, well, brk=self.get_breakpoints(X, y_hat), goal=goal)
            b = self.get_breakpoints(X, y_hat)
#            tst = set(item for item in b)
#            print(tst)
            return False,  b, X, y, y_hat, X_test, y_test#y_hat

    
# =============================================================================
#     generates training, validation and test sets from "data"
# =============================================================================
    def generate_sets(self, data, train_frac, val_frac):
        train_set, val_set = [], []
        train_size = int(np.round(train_frac * len(data)))
        val_size = int(np.round(val_frac * len(data)))
        while (len(train_set) < train_size):
            train_set.append(data.pop(r.randint(0, len(data)-1)))
        while (len(val_set) < val_size):
            val_set.append(data.pop(r.randint(0, len(data)-1)))
    ##    print("Train: ", len(train_set))
    ##    print("Val: ", len(val_set))
    ##    print("Test: ", len(data))
        train_set.sort()
        val_set.sort()
        data.sort()
        return train_set, val_set, data
    
# =============================================================================
#     retrieves breakpoints from a set of trained X and y values
# =============================================================================
    def get_breakpoints(self, X, y_hat):
        brk = []
        brk.append([X[0], y_hat[0]])
#        record = True
        for bf in self.model.basis_:
##            print(type(bf))
            if type(bf) is pyearth._basis.HingeBasisFunction:
                
                if(not bf.is_pruned()):    
                    to_check = bf.get_knot()
                    already = False
#                    print("ho",brk, to_check)
                    for b in brk:
                        if to_check in b:
                            already = True
                    if not already:
                        brk.append([bf.get_knot(), self.model.predict([bf.get_knot(),])[0]])
#                record = not record
        brk.append([X[-1], y_hat[-1]])

        return sorted(brk, key=lambda x: x[0])

#    def get_multi_breakpoints(self):
#        brk = []
#        for bf in self.model.basis_:
###            print(type(bf))
#            if type(bf) is pyearth._basis.HingeBasisFunction:
#                if(not bf.is_pruned()):
##                    print(bf.get_knot())
#                    #brk.append([bf.get_knot(), self.model.predict([bf.get_knot(),])[0]])
#        return brk
# =============================================================================
#     tests if the well is capable of being representet in 2D domain space
# =============================================================================
    def test3d(self, d_dict, inters, well):
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
        plotter.plot3d(t_x, t_y, [m[0] for m in t_z], well)


# =============================================================================
# run training for all the wells
# =============================================================================
def run_all():
    m = Mars()
    wells = m.df.well.unique()
    for well in wells:
        for goal in ['oil', 'gas']:
            run(well, normalize=False, goal=goal)

# =============================================================================
# records average validation set error for combinations of user specified parameters.
# Used for finding optimal parameters, cross-validation            
# =============================================================================
def test_error_run(well, iterations, test_frac, goal='oil', prune=False, max_terms=15, plot=False, normalize=False, hp=1, allow_nan = False, nan_ratio=0.3, intervals=100, factor=1.5, grid_size=10, verbose=0):
    penalties = [1,3,5,7,9,11,13,15,17]
    minspans = [1,3,5,7,9,11,13,15,17]
    errors = [[x for x in range(len(minspans))]for y in range(len(penalties))]
    for l in range(len(penalties)):
        for k in range(len(minspans)):
            print(l,k)
            error = 0
            m = Mars(prune, max_terms, penalties[l], minspans[k])
            for i in range(iterations):
#                if (i%50 == 0):
#                    print (i)
                _,_,_,_,_, X_test, y_test = m.run_well(well, goal, plot, normalize, hp, allow_nan, nan_ratio, intervals, factor, grid_size, verbose, 1-test_frac, 0)
                pred = m.model.predict(X_test)
                for j in range (len(X_test)):
                    error += abs(pred[j]-y_test[j][0])
            errors[l][k] = error/float(iterations)
            del(m)
    return errors

# =============================================================================
# calculate test error
# =============================================================================
def test_error_run_single(well, iterations, test_frac, penalty, minspan, goal='oil', prune=False, max_terms=15, plot=False, normalize=False, hp=1, allow_nan = False, nan_ratio=0.3, intervals=100, factor=1.5, grid_size=10, verbose=0):

    error = 0
    m = Mars(prune, max_terms, penalty, minspan)
    for i in range(iterations):
#                if (i%50 == 0):
#                    print (i)
        _,_,_,_,_, X_test, y_test = m.run_well(well, goal, plot, normalize, hp, allow_nan, nan_ratio, intervals, factor, grid_size, verbose, 1-test_frac, 0)
        pred = m.model.predict(X_test)
        for j in range (len(X_test)):
            error += abs(pred[j]-y_test[j][0])
    return error/float(iterations)
        
            
            
# =============================================================================
# runs a single well without creating an object first
# =============================================================================
def run(well, goal='oil',train_frac=1.0, val_frac=0.0, prune=False, max_terms=15, penalty=10, minspan=1, plot=True, normalize=False, hp=1, allow_nan = False, nan_ratio=0.3, intervals=100, factor=1.5, grid_size=10, verbose=0):
    m = Mars(prune, max_terms, penalty, minspan)
    a,b,X,y,pred = m.run_well(well, goal, plot, normalize, hp, allow_nan, nan_ratio, intervals, factor, grid_size, verbose, train_frac, val_frac)
    print("X:",X)
    print("y:",y)
    print("pred:",pred)
    #        if('choke' in d_dict.keys()):
#            self.plot_fig(X, y, y_hat, well, brk=self.get_multi_breakpoints())
#            self.test3d(d_dict, 15)
##            plotter.mesh(d_dict['gaslift'], d_dict['choke'], d_dict['output'])

#        else:
#            self.plot_fig(X, y, y_hat, well, brk=self.get_breakpoints())


