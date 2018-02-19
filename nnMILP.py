# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 12:34:20 2018

@author: bendiw
"""

from gurobipy import *
import numpy as np
import tens
import math




phasenames = ["oil", "gas"]
sepnames = ["LP", "HP"]
OIL = 0
GAS = 1
separator_dict = {"LP":2, "HP":1}

wellnames = ["A2", "A3", "A5", "A6", "A7", "A8", "B1", "B2", 
             "B3", "B4", "B5", "B6", "B7", "C1", "C2", "C3", "C4"]


well_to_sep = {"A2" : ["HP"], "A3": ["HP"], "A5": ["HP"], "A6": ["HP"], "A7": ["HP"], "A8": ["HP"], 
               "B1" : ["HP", "LP"], "B2" : ["HP", "LP"], "B3" : ["HP", "LP"], "B4" : ["HP", "LP"], "B5" : ["HP", "LP"], "B6" : ["HP", "LP"], "B7" : ["HP", "LP"], 
               "C1" : ["LP"], "C2" : ["LP"], "C3" : ["LP"], "C4" : ["LP"]}
platforms = ["A", "B", "C"]
p_dict = {"A" : ["A2", "A3", "A5", "A6", "A7", "A8"], "B":["B1", "B2", 
             "B3", "B4", "B5", "B6", "B7"], "C":["C1", "C2", "C3", "C4"]}
p_sep_route = {"A":[1], "B":[0,1], "C":[0]}
p_sep_names = {"A":["HP"], "B":["LP", "HP"], "C":["LP"]}
sep_p_route = {"LP": ["B", "C"], "HP":["A", "B"]}

# =============================================================================
# initialize an optimization model
# =============================================================================
def init(mode):
    pass

# =============================================================================
# call once model has been properly initialized
# =============================================================================
def run():
    pass

# =============================================================================
# load pre-trained neural nets from file
# =============================================================================
def loadNeuralNets():
    pass

# =============================================================================
# train a new neural net for a given well
# =============================================================================
def trainNeuralNet(well):
    pass

# =============================================================================
# main function to call from console
# =============================================================================
def opt(verbose=0):
    pass