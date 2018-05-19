# -*- coding: utf-8 -*-
"""
Created on Wed May 16 09:35:07 2018

@author: bendiw
"""
from gurobipy import *
import numpy as np
import math
import pandas as pd
import tools as t
import os.path
from recourse_models import NN, SOS2, Factor
   

# =============================================================================
# The recourse algorithm function
# =============================================================================
def recourse(num_iter, max_changes=3, init_name=None, model_type="sos2"):
    #our model
    model = get_model(model_type).init(?)
    
    #get initial solution
    init_sol = iteration(model, changes=max_changes, initial_flag=True)
    
    for i in range(num_iter):
        #TODO: iterate
        pass
    pass

# =============================================================================
# Perform one iteration of algorithm. initial_flag controls whether iteration
# should terminate after evaluating for the maximum number of changes and
# return solution - this is because all models will find same solution
# before learning any true well models.
# =============================================================================
def iteration(model, changes, initial_flag=False):
    #load "true" sos2 or NN models
    true_wells = t.?
    #TODO: load
    
    
# =============================================================================
# Evalate a solution given the true well curves and check infeasibility
# =============================================================================
def eval_sol(true_wells, chokes, indiv_cap, max_cap):
    pass
    
# =============================================================================
# helper function
# =============================================================================
def get_model(m_type):
    if(m_type=="sos2"):
        return SOS2()
    elif(m_type=="nn"):
        return NN()
    elif(m_type=="factor"):
        return Factor()
    else:
        raise ValueError("No model type specified!")
    