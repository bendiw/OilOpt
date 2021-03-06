# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 15:08:13 2018

@author: bendi
"""

# -*- coding: utf-8 -*f-

# =============================================================================
# Nomenclature:
#   Thesis  |   Code
#   x       |    mu
#   s       |    rho
#   z       |    lambda
# =============================================================================
"""
Created on Wed Feb 21 10:33:12 2018

@author: bendiw
"""
from gurobipy import *
import numpy as np
import math
import pandas as pd
import tools as t
import os.path
from recourse_model import Recourse_Model

class Factor(Recourse_Model):

    # =============================================================================
    # get neural nets either by loading existing ones or training new ones
    # =============================================================================
    def getNeuralNets(self, mode, case, net_type="mean"):
        weights = {well : {} for well in self.wellnames}
        biases = {well : {} for well in self.wellnames}
        multidims = {well : {} for well in self.wellnames}
        layers = {well : {} for well in self.wellnames}
        for well in self.wellnames:
            for phase in self.phasenames:
                weights[well][phase] = {}
                biases[well][phase] = {}
                multidims[well][phase] = {}
                layers[well][phase] = {} 
                for separator in self.well_to_sep[well]:
                    if mode==self.LOAD:
                        multidims[well][phase][separator], weights[well][phase][separator], biases[well][phase][separator] = t.load_2(well, phase, separator, case, net_type)
                        layers[well][phase][separator] = len(multidims[well][phase][separator])
                        if net_type=="mean" and multidims[well][phase][separator][layers[well][phase][separator]-1] > 1:
                            multidims[well][phase][separator][layers[well][phase][separator]-1] -=1
                    else:
                        layers[well][phase][separator], multidims[well][phase][separator], weights[well][phase][separator], biases[well][phase][separator] = t.train(well, phase, separator, case)
        return layers, multidims, weights, biases
    
        
 
    # =============================================================================
    # Function to run with all wells in the problem.
    # =============================================================================
    def init(self, case=2,
                num_scen = 10, lower=-4, upper=4, phase="gas", sep="HP", save=True,store_init=False, init_name=None, 
                max_changes=15, w_relative_change=None, stability_iter=None, distr="truncnorm", lock_wells=None, scen_const=None, recourse_iter=False, verbose=1):
        
        Recourse_Model.init(self, case, num_scen, lower, upper, phase, sep, save, store_init, init_name, max_changes, w_relative_change, stability_iter, distr, lock_wells, scen_const, recourse_iter, verbose)        

        
        self.results_file = "results/robust/res_factor.csv"
        self.s_draw = t.get_scenario(case, num_scen, lower=lower, upper=upper,
                                     phase=phase, sep=sep, iteration=stability_iter, distr=distr)
        
        #load mean and variance networks
        self.layers, self.multidims, self.weights, self.biases = self.getNeuralNets(self.LOAD, case, net_type="mean")
        self.layers_var, self.multidims_var, self.weights_var, self.biases_var = self.getNeuralNets(self.LOAD, case, net_type="std")
        
        # =============================================================================
        # variable creation                    
        # =============================================================================
        self.lambdas = self.m.addVars([(well,phase,sep, layer, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers[well][phase][sep]-1) for neuron in range(self.multidims[well][phase][sep][layer])], vtype = GRB.BINARY, name="lambda")
        self.mus = self.m.addVars([(well,phase,sep,layer, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers[well][phase][sep]-1) for neuron in range(self.multidims[well][phase][sep][layer])], vtype = GRB.CONTINUOUS, name="mu")
        self.rhos = self.m.addVars([(well,phase,sep,layer, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers[well][phase][sep]-1) for neuron in range(self.multidims[well][phase][sep][layer])], vtype = GRB.CONTINUOUS, name="rho")

        #variance networks
        self.lambdas_var = self.m.addVars([(well,phase,sep, layer, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1,self.layers_var[well][phase][sep]) for neuron in range(self.multidims_var[well][phase][sep][layer])], vtype = GRB.BINARY, name="lambda_var")
        self.mus_var = self.m.addVars([(well,phase,sep,layer, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1,self.layers_var[well][phase][sep]) for neuron in range(self.multidims_var[well][phase][sep][layer])], vtype = GRB.CONTINUOUS, name="mu_var")
        self.rhos_var = self.m.addVars([(well,phase,sep,layer, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1,self.layers_var[well][phase][sep]) for neuron in range(self.multidims_var[well][phase][sep][layer])], vtype = GRB.CONTINUOUS, name="rho_var")


        # =============================================================================
        # NN MILP constraints creation. Mean and variance networks.
        # Variables mu and rho are indexed from 1 and up since the input layer
        # is layer 0 in multidims. 
        # =============================================================================
        # mean
        self.m.addConstrs(self.mus[well, phase, sep, 1, neuron] - self.rhos[well, phase, sep, 1, neuron] - quicksum(self.weights[well][phase][sep][0][dim][neuron]*self.inputs[well, sep, dim] for dim in range(self.multidims[well][phase][sep][0])) == self.biases[well][phase][sep][0][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(self.multidims[well][phase][sep][1]) )
        self.m.addConstrs(self.mus[well, phase, sep, layer, neuron] - self.rhos[well, phase, sep, layer, neuron] - quicksum((self.weights[well][phase][sep][layer-1][dim][neuron]*self.mus[well, phase, sep, layer-1, dim]) for dim in range(self.multidims[well][phase][sep][layer-1])) == self.biases[well][phase][sep][layer-1][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(2, self.layers[well][phase][sep]-1) for neuron in range(self.multidims[well][phase][sep][layer]) )
        # var
        self.m.addConstrs(self.mus_var[well, phase, sep, 1, neuron] - self.rhos_var[well, phase, sep, 1, neuron] - quicksum(self.weights_var[well][phase][sep][0][dim][neuron]*self.inputs[well, sep, dim] for dim in range(self.multidims_var[well][phase][sep][0])) == self.biases_var[well][phase][sep][0][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(self.multidims_var[well][phase][sep][1]) )
        self.m.addConstrs(self.mus_var[well, phase, sep, layer, neuron] - self.rhos_var[well, phase, sep, layer, neuron] - quicksum((self.weights_var[well][phase][sep][layer-1][dim][neuron]*self.mus_var[well, phase, sep, layer-1, dim]) for dim in range(self.multidims_var[well][phase][sep][layer-1])) == self.biases_var[well][phase][sep][layer-1][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(2, self.layers_var[well][phase][sep]-1) for neuron in range(self.multidims_var[well][phase][sep][layer]) )

        #indicator constraints
        #mean
        self.m.addConstrs( (self.lambdas[well, phase, sep, layer, neuron] == 1) >> (self.mus[well, phase, sep, layer, neuron] <= 0)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers[well][phase][sep]-1) for neuron in range(self.multidims[well][phase][sep][layer]))
        self.m.addConstrs( (self.lambdas[well, phase, sep, layer, neuron] == 0) >> (self.rhos[well, phase, sep, layer, neuron] <= 0)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1,self.layers[well][phase][sep]-1) for neuron in range(self.multidims[well][phase][sep][layer]))
        #variance
        self.m.addConstrs( (self.lambdas_var[well, phase, sep, layer, neuron] == 1) >> (self.mus_var[well, phase, sep, layer, neuron] <= 0)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_var[well][phase][sep]-1) for neuron in range(self.multidims_var[well][phase][sep][layer]))
        self.m.addConstrs( (self.lambdas_var[well, phase, sep, layer, neuron] == 0) >> (self.rhos_var[well, phase, sep, layer, neuron] <= 0)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_var[well][phase][sep]-1) for neuron in range(self.multidims_var[well][phase][sep][layer]))

        #use these to model last layer as linear instead of ReLU
        #gas
        self.m.addConstrs( (self.routes[well, sep] == 1) >> (self.outputs_gas[scenario, well, sep] ==  quicksum(self.weights[well]["gas"][sep][self.layers[well]["gas"][sep]-2][neuron][0] * self.mus[well, "gas", sep, self.layers[well]["gas"][sep]-2, neuron] for neuron in range(self.multidims[well]["gas"][sep][self.layers[well]["gas"][sep]-2]) ) + self.biases[well]["gas"][sep][self.layers[well]["gas"][sep]-2][0]  + 
                          self.s_draw.loc[scenario][well] *(quicksum(self.weights_var[well]["gas"][sep][self.layers_var[well]["gas"][sep]-2][neuron][0] * (self.mus_var[well, "gas", sep, self.layers_var[well]["gas"][sep]-2, neuron]) for neuron in range(self.multidims_var[well]["gas"][sep][self.layers_var[well]["gas"][sep]-2])) + self.biases_var[well]["gas"][sep][self.layers_var[well]["gas"][sep]-2][0]) )  for well in self.wellnames for sep in self.well_to_sep[well] for scenario in range(self.scenarios))
        self.m.addConstrs( (self.routes[well, sep] == 0) >> (self.outputs_gas[scenario, well, sep] == 0) for well in self.wellnames for sep in self.well_to_sep[well] for scenario in range(self.scenarios))
        #oil
        self.m.addConstrs( (self.routes[well, sep] == 1) >> (self.outputs_oil[well, sep] == quicksum(self.weights[well]["oil"][sep][self.layers[well]["oil"][sep]-2][neuron][0] * self.mus[well, "oil", sep, self.layers[well]["oil"][sep]-2, neuron] for neuron in range(self.multidims[well]["oil"][sep][self.layers[well]["oil"][sep]-2]) ) + self.biases[well]["oil"][sep][self.layers[well]["oil"][sep]-2][0]) for well in self.wellnames for sep in self.well_to_sep[well])
        self.m.addConstrs( (self.routes[well, sep] == 0) >> (self.outputs_oil[well, sep] == 0) for well in self.wellnames for sep in self.well_to_sep[well] )
        
        
        return self
        
