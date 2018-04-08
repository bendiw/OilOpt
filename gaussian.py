# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:54:05 2018

@author: bendiw
"""

import GPy
import numpy as np
import caseloader as cl
import tens
#%matplotlib inline
from sklearn.preprocessing import normalize, RobustScaler

from matplotlib import pyplot as plt
from IPython.display import display
GPy.plotting.change_plotting_library('matplotlib')
#data = [[[x], [(x**2)+np.random.randint(0, 25000)]] for x in range(-300, 300)]
#X = np.array([x[0] for x in data])
#y = np.array([x[1] for x in data])

def load_well(well, separator, goal, hp, factor, intervals, nan_ratio):
    df = cl.load("welltests_new.csv")
    dict_data,_,_ = cl.gen_targets(df, well+"", goal=goal, normalize=False, intervals=intervals,
                               factor = factor, nan_ratio = nan_ratio, hp=hp) #,intervals=100
    data = tens.convert_from_dict_to_tflists(dict_data)
    return data

data = load_well("A3", "HP", "oil", 1, 1.5, 20, 0.3)
#data = [[[x], [x**2]] for x in range(-300, 300)]
if (len(data[0][0]) >= 2):
    is_3d = True
    dim = 2
else:
    is_3d = False
    dim=1
rs = RobustScaler(with_centering =False)
if is_3d:
    glift_orig = np.array([x[0][0] for x in data])
    choke_orig = np.array([x[0][1] for x in data])
    y_orig = np.array([x[1][0] for x in data]).reshape(-1,1)
    glift = rs.fit_transform(glift_orig.reshape(-1,1))
    choke = rs.transform(choke_orig.reshape(-1,1))
    y = rs.transform(y_orig.reshape(-2, 1))
    X =np.array([[glift[i][0], choke[i][0]] for i in range(len(glift))])
else:
    X_orig = np.array([x[0][0] for x in data]).reshape(-1,1)
    y_orig = np.array([x[1][0] for x in data]).reshape(-1,1)
    X = rs.fit_transform(X_orig.reshape(-1,1))
    y = rs.transform(y_orig.reshape(-1, 1))
#X = np.random.uniform(-3.,3.,(20,1))
#Y = np.sin(X) + np.random.randn(20,1)*0.05
#print(X)
#kernel = GPy.kern.PolynomialBasisFuncKernel(input_dim=1, degree=2)
kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=0.4) 
m = GPy.models.GPRegression(X,y,kernel)


m.optimize(messages=True)
display(m)
fig = m.plot()
GPy.plotting.show(fig)


