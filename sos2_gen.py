# -*- coding: utf-8 -*-
"""
Created on Tue May 15 08:34:28 2018

@author: arntgm
"""

import tools
import numpy as np
from matplotlib import pyplot


def hey(wells, phase="oil", mode="mean"):
    filename = "variance_case"+str(case)+"_"+goal+".csv"
    df = pd.read_csv(filename, sep=';', index_col=0)
    for well in wells:
        
    
    
    
    model = tools.retrieve_model(well, goal=phase, mode=mode)
    print(model.layers)
    X=np.array([[i] for i in range(101)])    
    prediction = [x[0] for x in model.predict(X)]
    
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
#    line1 = ax.plot(X, pre,color="green",linestyle="None", marker=".", markersize=5)
#    line3 = ax.plot(X, m, color="black", linewidth=.5)
    line2 = ax.plot(X, prediction, color='green',linestyle='dashed', linewidth=1)
    pyplot.xlabel('choke')
    pyplot.ylabel(goal)
#    pyplot.fill_between([x[0] for x in X], mean-std, mean+std,
#                       alpha=0.2, facecolor='#089FFF', linewidth=1)
#    pyplot.fill_between([x[0] for x in X], mean-2*std, mean+2*std,
#                       alpha=0.2, facecolor='#089FFF', linewidth=1)   