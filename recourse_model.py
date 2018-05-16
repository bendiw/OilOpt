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
class Recourse_Model:
    well_to_sep = {}
    wellnames = []
    platforms = []
    p_dict = {}
    phasenames = []
    p_sep_names ={}
    LOAD = 0
    OIL = 0
    GAS = 1
    LOAD = 0
    TRAIN = 1

    


    # =============================================================================
    # get neural nets either by loading existing ones or training new ones
    # =============================================================================
    def getNeuralNets(self, mode, case, phase, net_type="scen"):
        weights = {scenario : {well : {} for well in self.wellnames} for scenario in range(self.scenarios)}
        biases = {scenario : {well : {} for well in self.wellnames}for scenario in range(self.scenarios)}
        multidims = {scenario : {well : {} for well in self.wellnames} for scenario in range(self.scenarios) }
        layers = {scenario: {well : {} for well in self.wellnames}for scenario in range(self.scenarios)}
        for scenario in range(self.scenarios):
            for well in self.wellnames:
                weights[scenario][well] = {}
                biases[scenario][well] = {}
                multidims[scenario][well] = {}
                layers[scenario][well] = {} 
                for separator in self.well_to_sep[well]:
                    if mode==self.LOAD:
                        multidims[scenario][well][separator], weights[scenario][well][separator], biases[scenario][well][separator] = t.load_2(scenario, well, phase, separator, case, net_type)
                        layers[scenario][well][separator] = len(multidims[scenario][well][separator])
                        if net_type=="mean" and multidims[scenario][well][separator][ layers[scenario][well][separator]-1 ] > 1:
                            multidims[scenario][well][separator][ layers[scenario][well][separator]-1 ] -=1
                    else:
                        layers[scenario][well][separator], multidims[scenario][well][separator], weights[scenario][well][separator], biases[scenario][well][separator] = t.train(well, phase, separator, case)
        return layers, multidims, weights, biases
    
        
    
    
    # =============================================================================
    # Function to build a model with all wells in the problem.
    # =============================================================================
    def init(self, case=2,
                num_scen = 1000, lower=-4, upper=4, phase="gas", sep="HP", save=True,store_init=False, init_name=None, 
                max_changes=15, w_relative_change=None, stability_iter=None, distr="truncnorm", lock_wells=None, scen_const=None, recourse_iter=False, verbose=1):
        if(case==2):
            self.wellnames = t.wellnames_2
            self.well_to_sep = t.well_to_sep_2
        else:
            self.wellnames = t.wellnames
            self.well_to_sep = t.well_to_sep
            self.platforms= t.platforms
            self.p_dict = t.p_dict
            self.p_sep_names = t.p_sep_names
        
        self.store_init = store_init
        self.init_name = init_name
        self.case = case
        self.recourse_iter = recourse_iter
        self.max_changes = max_changes
        self.verbose=verbose
        self.save=save
#        self.s_draw = t.get_scenario(case, num_scen, lower=lower, upper=upper,
#                                     phase=phase, sep=sep, iteration=stability_iter, distr=distr)
        self.scenarios = num_scen
        
        #alternative results file for storing init solutions
        self.results_file_init = "results/initial/res_initial.csv"

        self.phasenames = t.phasenames
            
        #Case relevant numerics
        if case==2:
            if(not w_relative_change):
                self.w_relative_change = {well : [0.4] for well in self.wellnames}
            
            if(lock_wells):
                #wells which are not allowed to change
                assert(isinstance(lock_wells, (list)))
                for w in lock_wells:
                    self.w_relative_change[w] = [0, 0]
                    
            if(scen_const):
                #provide "locked" scenario constant for certain wells
                assert(isinstance(scen_const, (dict)))
                for well, val in scen_const.items():
                    self.s_draw[well] = val
            
            if not init_name:
                self.w_initial_prod = {well : 0 for well in self.wellnames}
                self.w_initial_vars = {well : [0] for well in self.wellnames}
            elif not isinstance(init_name, (dict)):
                #load init from file
                self.w_initial_df = t.get_robust_solution(init_name=init_name)
                self.w_initial_vars = {w:[w_initial_df[w+"_choke"].values[0]] for w in self.wellnames}
                print(self.w_initial_vars)
                
                self.w_initial_prod = {well : 1. if self.w_initial_vars[well][0]>0 else 0. for well in self.wellnames}
                print(self.w_initial_prod)
            #constraints for case 2
            else:
                #we were given a dict of initial values
#                w_initial_vars=init_name
                self.w_initial_vars={}
                for w in self.wellnames:
                    self.w_initial_vars[w] = [init_name[w+"_choke"]]
#                    del w_initial_vars[w+"_choke"]
                
#                print("optimization initial chokes:", w_initial_vars)
                self.w_initial_prod = {well : 1 if self.w_initial_vars[well][0]>0 else 0 for well in self.wellnames}
            self.tot_exp_cap = 250000
            self.well_cap = 54166
        else:
            raise ValueError("Case 1 not implemented yet.")
            
        # =============================================================================
        # initialize an optimization model
        # =============================================================================
        self.m = Model("Model")

            
        # =============================================================================
        # get input variable bounds        
        # =============================================================================
        if(case==1):
            w_min_glift, w_max_glift = t.get_limits("gaslift_rate", self.wellnames, self.well_to_sep, case)
        else:
            w_min_glift = None
            w_max_glift = None
        w_min_choke, w_max_choke = t.get_limits("choke", self.wellnames, self.well_to_sep, case)
        
        #NOTE: changed order of inputs. needs to be taken into consideration when impl. case 1
        w_max_lims = [w_max_choke, w_max_glift] 
        w_min_lims = [w_min_choke, w_min_glift]
        self.input_upper = {(well, sep, dim) : w_max_lims[dim][well][sep]  for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1)}
        self.input_lower = {(well, sep, dim) : w_min_lims[dim][well][sep]  for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1)}
        
        #new variables to control routing decision and input/output
        self.outputs_gas = self.m.addVars([(scenario, well, sep) for well in self.wellnames for sep in self.well_to_sep[well] for scenario in range(self.scenarios)], vtype = GRB.CONTINUOUS, name="outputs_gas")
        self.outputs_oil = self.m.addVars([(well, sep) for well in self.wellnames for sep in self.well_to_sep[well]], vtype = GRB.CONTINUOUS, name="outputs_oil")
        self.routes = self.m.addVars([(well, sep) for well in self.wellnames for sep in self.well_to_sep[well]], vtype = GRB.BINARY, name="routing")
        self.changes = self.m.addVars([(well, sep, dim) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1)], vtype=GRB.BINARY, name="changes")
        self.inputs = self.m.addVars(self.input_upper.keys(), ub = self.input_upper, lb=self.input_lower, name="input", vtype=GRB.SEMICONT) #SEMICONT
        
        
        
        #SUBCLASS DEPENDENT CODE
# =============================================================================
#         <<<<<<<<<<<<<<
#         #load mean and variance networks
#         self.layers_gas, self.multidims_gas, self.weights_gas, self.biases_gas = self.getNeuralNets(self.LOAD, case, net_type="scen", phase="gas")
#         self.layers_oil, self.multidims_oil, self.weights_oil, self.biases_oil = self.getNeuralNets(self.LOAD, case, net_type="scen", phase="oil")
#         
#         
#         # =============================================================================
#         # variable creation                    
#         # Gas networks scenario dependent, oil networks are not
#         # =============================================================================
#         lambdas_gas = self.m.addVars([(scenario, well,sep, layer, neuron) for scenario in range(self.scenarios) for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers[well][phase][sep]-1) for neuron in range(self.multidims[well][phase][sep][layer])], vtype = GRB.BINARY, name="lambda_gas")
#         mus_gas = self.m.addVars([(scenario, well,sep,layer, neuron) for scenario in range(self.scenarios)  for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers[well][phase][sep]-1) for neuron in range(self.multidims[well][phase][sep][layer])], vtype = GRB.CONTINUOUS, name="mu_gas")
#         rhos_gas = self.m.addVars([(scenario, well,sep,layer, neuron) for scenario in range(self.scenarios) for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers[well][phase][sep]-1) for neuron in range(self.multidims[well][phase][sep][layer])], vtype = GRB.CONTINUOUS, name="rho_gas")
#         
#         lambdas_oil = self.m.addVars([(well,sep, layer, neuron) for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers[well][phase][sep]-1) for neuron in range(self.multidims[well][phase][sep][layer])], vtype = GRB.BINARY, name="lambda_oil")
#         mus_oil = self.m.addVars([(well,sep,layer, neuron)  for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers[well][phase][sep]-1) for neuron in range(self.multidims[well][phase][sep][layer])], vtype = GRB.CONTINUOUS, name="mu_oil")
#         rhos_oil = self.m.addVars([(well,sep,layer, neuron)   for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers[well][phase][sep]-1) for neuron in range(self.multidims[well][phase][sep][layer])], vtype = GRB.CONTINUOUS, name="rho_oil")
# 
#         
# 
#         # =============================================================================
#         # NN MILP constraints creation. Mean and variance networks.
#         # Variables mu and rho are indexed from 1 and up since the input layer
#         # is layer 0 in multidims. 
#         # =============================================================================
# 
#         # gas
#         self.m.addConstrs(mus_gas[scenario, well, phase, sep, 1, neuron] - rhos_gas[scenario, well, phase, sep, 1, neuron] - quicksum(self.weights_gas[scenario][well][phase][sep][0][dim][neuron]*inputs[well, sep, dim] for dim in range(self.multidims_gas[scenario][well][phase][sep][0])) == self.biases_gas[scenario][well][phase][sep][0][neuron] for scenario in range(self.scenarios) for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(self.multidims_gas[scenario][well][phase][sep][1]))
#         self.m.addConstrs(mus_gas[scenario, well, phase, sep, layer, neuron] - rhos_gas[scenario, well, phase, sep, layer, neuron] - quicksum((self.weights_gas[scenario][well][phase][sep][layer-1][dim][neuron]*mus_gas[scenario, well, phase, sep, layer-1, dim]) for dim in range(self.multidims_gas[scenario][well][phase][sep][layer-1])) == self.biases_gas[scenario][well][phase][sep][layer-1][neuron] for scenario in range(self.scenarios) for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(2, self.layers_gas[scenario][well][phase][sep]-1) for neuron in range(self.multidims_gas[scenario][well][phase][sep][layer]) )
#        
#         # oil
#         self.m.addConstrs(mus_oil[well, phase, sep, 1, neuron] - rhos_oil[well, phase, sep, 1, neuron] - quicksum(self.weights_oil[well][phase][sep][0][dim][neuron]*inputs[well, sep, dim] for dim in range(self.multidims_oil[well][phase][sep][0])) == self.biases_oil[well][phase][sep][0][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(self.multidims_oil[well][phase][sep][1]) )
#         self.m.addConstrs(mus_oil[well, phase, sep, layer, neuron] - rhos_oil[well, phase, sep, layer, neuron] - quicksum((self.weights_oil[well][phase][sep][layer-1][dim][neuron]*mus_oil[well, phase, sep, layer-1, dim]) for dim in range(self.multidims_oil[well][phase][sep][layer-1])) == self.biases_oil[well][phase][sep][layer-1][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(2, self.layers_oil[well][phase][sep]-1) for neuron in range(self.multidims_oil[well][phase][sep][layer]) )
# 
# 
# #        self.m.addConstrs(mus_var[well, phase, sep, 1, neuron] - rhos_var[well, phase, sep, 1, neuron] - quicksum(self.weights_var[well][phase][sep][0][dim][neuron]*inputs[well, sep, dim] for dim in range(self.multidims_var[well][phase][sep][0])) == self.biases_var[well][phase][sep][0][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(self.multidims_var[well][phase][sep][1]) )
# #        self.m.addConstrs(mus_var[well, phase, sep, layer, neuron] - rhos_var[well, phase, sep, layer, neuron] - quicksum((self.weights_var[well][phase][sep][layer-1][dim][neuron]*mus_var[well, phase, sep, layer-1, dim]) for dim in range(self.multidims_var[well][phase][sep][layer-1])) == self.biases_var[well][phase][sep][layer-1][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(2, self.layers_var[well][phase][sep]) for neuron in range(self.multidims_var[well][phase][sep][layer]) )
#         #indicator constraints
#         if(load_M):
#             pass
# #            self.m.addConstrs(mus[well, phase, sep, layer, neuron] <= (big_M[well][phase][sep][neuron][0])*(1-lambdas[well, phase, sep, layer, neuron]) for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers[well][phase][sep]) for neuron in range(len(self.biases[well][phase][sep][0])))
# #            self.m.addConstrs(rhos[well, phase, sep, layer, neuron] <= (big_M[well][phase][sep][neuron][1])*(lambdas[well, phase, sep, layer, neuron]) for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers[well][phase][sep]) for neuron in range(len(self.biases[well][phase][sep][0])))
#         else:
#             #gas
#             self.m.addConstrs( (lambdas_gas[scenario, well, phase, sep, layer, neuron] == 1) >> (mus_gas[scenario, well, phase, sep, layer, neuron] <= 0) for scenario in range(self.scenarios)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_gas[scenario][well][phase][sep]-1) for neuron in range(self.multidims_gas[scenario][well][phase][sep][layer]))
#             self.m.addConstrs( (lambdas_gas[scenario, well, phase, sep, layer, neuron] == 0) >> (rhos_gas[scenario, well, phase, sep, layer, neuron] <= 0) for scenario in range(self.scenarios) for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1,self.layers_gas[scenario][well][phase][sep]-1) for neuron in range(self.multidims_gas[scenario][well][phase][sep][layer]))
# 
#             #oil
#             self.m.addConstrs( (lambdas_oil[well, phase, sep, layer, neuron] == 1) >> (mus_oil[well, phase, sep, layer, neuron] <= 0)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_oil[well][phase][sep]-1) for neuron in range(self.multidims_oil[well][phase][sep][layer]))
#             self.m.addConstrs( (lambdas_oil[well, phase, sep, layer, neuron] == 0) >> (rhos_oil[well, phase, sep, layer, neuron] <= 0)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_oil[well][phase][sep]-1) for neuron in range(self.multidims_oil[well][phase][sep][layer]))
# 
# #            self.m.addConstrs( (lambdas_var[well, phase, sep, layer, neuron] == 1) >> (mus_var[well, phase, sep, layer, neuron] <= 0)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_var[well][phase][sep]) for neuron in range(self.multidims_var[well][phase][sep][layer]))
# #            self.m.addConstrs( (lambdas_var[well, phase, sep, layer, neuron] == 0) >> (rhos_var[well, phase, sep, layer, neuron] <= 0)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_var[well][phase][sep]) for neuron in range(self.multidims_var[well][phase][sep][layer]))
# 
# 
#         # =============================================================================
#         # big-M constraints on output/routing        
#         # since we model output neurons as ReLU, these are simple. may change later so that output are purely linear
#         # =============================================================================
#         #gas output
# #        self.m.addConstrs( (routes[well, sep] == 1) >> (outputs_gas[scenario, well, "gas", sep] == mus[well, "gas", sep, self.layers[well]["gas"][sep]-1, neuron] + self.s_draw.loc[scenario][well]*mus_var[well, "gas", sep, self.layers_var[well]["gas"][sep]-1, neuron])  for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(self.multidims[well]["gas"][sep][-1]) for scenario in range(self.scenarios))
# #        self.m.addConstrs( (routes[well, sep] == 0) >> (outputs_gas[scenario, well, "gas", sep] == 0) for well in self.wellnames for sep in self.well_to_sep[well] for scenario in range(self.scenarios))
#     
#         #oil output
# #        self.m.addConstrs( (routes[well, sep] == 1) >> (outputs_oil[well, "oil", sep] == mus[well, "oil", sep, self.layers[well]["oil"][sep]-1, neuron])  for well in self.wellnames for "oil" in self.phasenames for sep in self.well_to_sep[well] for neuron in range(self.multidims_var[well]["oil"][sep][-1]))
# #        self.m.addConstrs( (routes[well, sep] == 1) >> (outputs_oil[well, "oil", sep] == mus[well, "oil", sep, self.layers[well]["oil"][sep]-1, neuron])  for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(self.multidims_var[well]["oil"][sep][-1]))
# #        self.m.addConstrs( (routes[well, sep] == 0) >> (outputs_oil[well, "oil", sep] == 0) for well in self.wellnames for sep in self.well_to_sep[well] )
#         
#         
#         #use these to model last layer as linear instead of ReLU
#         #gas
#         self.m.addConstrs( (routes[well, sep] == 1) >> (outputs_gas[scenario, well, sep] ==  quicksum(self.weights_gas[scenario][well]["gas"][sep][self.layers_gas[scenario][well]["gas"][sep]-2][neuron][0] * mus_gas[scenario, well, "gas", sep, self.layers_gas[scenario][well]["gas"][sep]-2, neuron] for neuron in range(self.multidims_gas[scenario][well]["gas"][sep][self.layers_gas[scenario][well]["gas"][sep]-2]) ) + self.biases_gas[scenario][well]["gas"][sep][self.layers_gas[scenario][well]["gas"][sep]-2][0] for scenario in range(self.scenarios) for well in self.wellnames for sep in self.well_to_sep[well]))
#         self.m.addConstrs( (routes[well, sep] == 0) >> (outputs_gas[scenario, well, sep] == 0) for scenario in range(self.scenarios) for well in self.wellnames for sep in self.well_to_sep[well] for scenario in range(self.scenarios))
#     
#         #oil
#         self.m.addConstrs( (routes[well, sep] == 1) >> (outputs_oil[well, sep] == quicksum(self.weights_oil[well]["oil"][sep][self.layers_oil[well]["oil"][sep]-2][neuron][0] * mus_oil[well, "oil", sep, self.layers_oil[well]["oil"][sep]-2, neuron] for neuron in range(self.multidims_oil[well]["oil"][sep][self.layers_oil[well]["oil"][sep]-2]) ) + self.biases_oil[well]["oil"][sep][self.layers_oil[well]["oil"][sep]-2][0]) for well in self.wellnames for sep in self.well_to_sep[well])
#         self.m.addConstrs( (routes[well, sep] == 0) >> (outputs_oil[well, sep] == 0) for well in self.wellnames for sep in self.well_to_sep[well] )
#         
#         <<<<<<<<<<<<<<<<<<<<<<<<<<<<
# =============================================================================
        
        
        
        
        
        
        
        #input forcing to track changes
        self.m.addConstrs( (self.routes[well, sep] == 0) >> (self.inputs[well, sep, dim] <= 0) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1))

        
        # =============================================================================
        # separator gas constraints
        # =============================================================================
        gas_constr = self.m.addConstrs(self.outputs_gas[scenario, well, "HP"] <= self.well_cap for well in self.wellnames for scenario in range(self.scenarios))
            
        exp_constr = self.m.addConstrs(quicksum(self.outputs_gas[scenario, well, sep] for well in self.wellnames for sep in self.well_to_sep[well]) <= self.tot_exp_cap for scenario in range(self.scenarios))
        
        # =============================================================================
        # routing
        # =============================================================================
        self.m.addConstrs(quicksum(self.routes[well, sep] for sep in self.well_to_sep[well]) <= 1 for well in self.wellnames)

        # =============================================================================
#         change tracking and total changes
#         =============================================================================
        self.m.addConstrs(self.w_initial_vars[well][dim] - self.inputs[well, sep, dim] <= self.changes[well, sep, dim]*self.w_initial_vars[well][dim]*self.w_relative_change[well][dim] + 
                          (self.w_initial_vars[well][dim]*(1-quicksum(self.routes[well, separ] for separ in self.well_to_sep[well]))*(1-self.w_relative_change[well][dim])) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1))
        
        self.m.addConstrs(self.inputs[well, sep, dim] - self.w_initial_vars[well][dim] <= self.changes[well, sep, dim]*self.w_initial_vars[well][dim]*self.w_relative_change[well][dim]+
                          (1-self.w_initial_prod[well])*w_max_lims[dim][well][sep]*self.changes[well, sep, dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1))
        
        self.m.addConstr(quicksum(self.changes[well, sep, dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1)) <= self.max_changes)




    def solve(self):
        # =============================================================================
        # Solver parameters
        # =============================================================================
#        self.m.setParam(GRB.Param.NumericFocus, 2)
        self.m.setParam(GRB.Param.LogToConsole, self.verbose)
#        self.m.setParam(GRB.Param.Heuristics, 0)
#        self.m.setParam(GRB.Param.Presolve, 0)
#        self.m.Params.timeLimit = 360.0
#        self.m.setParam(GRB.Param.LogFile, "log.txt")
        self.m.setParam(GRB.Param.DisplayInterval, 15.0)
        
        #maximization of mean oil. no need to take mean over scenarios since only gas is scenario dependent
        self.m.setObjective( quicksum(self.outputs_oil[well, sep] for well in self.wellnames for sep in self.well_to_sep[well]), GRB.MAXIMIZE)
        self.m.optimize()
        
#    def get_solution(self):
#        df = pd.DataFrame(columns=t.robust_res_columns) 
#        chokes = [inputs[well, "HP", 0].x if outputs_gas[0, well, "HP"].x>0 else 0 for well in self.wellnames]
#        rowlist=[]
#        if(case==2 and save):
#            gas_mean = np.zeros(len(self.wellnames))
#            gas_var=[]
#            w = 0
#            for well in self.wellnames:
#                for scenario in range(self.scenarios):
#                    gas_mean[w] += outputs_gas[scenario, well, "HP"].x
#                gas_var.append(sum(self.weights_var[well]["gas"]["HP"][self.layers_var[well]["gas"]["HP"]-2][neuron][0] * (mus_var[well, "gas", "HP", self.layers_var[well]["gas"]["HP"]-2, neuron].x) for neuron in range(self.multidims_var[well]["gas"]["HP"][self.layers_var[well]["gas"]["HP"]-2])) + self.biases_var[well]["gas"]["HP"][self.layers_var[well]["gas"]["HP"]-2][0])
#
#                w += 1
#            gas_mean = (gas_mean/float(self.scenarios)).tolist()
#            oil_mean = [outputs_oil[well, "HP"].x for well in self.wellnames]
##            oil_var = [outputs_]
#            tot_oil = sum(oil_mean)
#            tot_gas = sum(gas_mean)
#            change = [abs(changes[w, "HP", 0].x) for w in self.wellnames]
##            print(df.columns)
#            rowlist = [self.scenarios, tot_exp_cap, well_cap, tot_oil, tot_gas]+chokes+gas_mean+oil_mean+gas_var+change
##            print(len(rowlist))
#            if(store_init):
#                df.rename(columns={"scenarios": "name"}, inplace=True)
#                rowlist[0] = init_name
#                df.loc[df.shape[0]] = rowlist
#
#                head = not os.path.isfile(self.results_file_init)
#                with open(self.results_file_init, 'a') as f:
#                    df.to_csv(f, sep=';', index=False, header=head)
#            else:
#                df.loc[df.shape[0]] = rowlist
#                head = not os.path.isfile(self.results_file)
#                with open(self.results_file, 'a') as f:
#                    df.to_csv(f, sep=';', index=False, header=head)
#        elif recourse_iter:
#            oil_mean = [outputs_oil[well, "HP"].x for well in self.wellnames]
#            gas_mean = []
#            gas_var = []
#            for well in self.wellnames:
#                gas_mean.append(sum(self.weights[well]["gas"]["HP"][self.layers[well]["gas"]["HP"]-2][neuron][0] * mus[well, "gas", "HP", self.layers[well]["gas"]["HP"]-2, neuron].x for neuron in range(self.multidims[well]["gas"]["HP"][self.layers[well]["gas"]["HP"]-2]) ) + self.biases[well]["gas"]["HP"][self.layers[well]["gas"]["HP"]-2][0] if outputs_gas[0, well, "HP"].x>0 else 0)
#                gas_var.append(sum(self.weights_var[well]["gas"]["HP"][self.layers_var[well]["gas"]["HP"]-2][neuron][0] * (mus_var[well, "gas", "HP", self.layers_var[well]["gas"]["HP"]-2, neuron].x) for neuron in range(self.multidims_var[well]["gas"]["HP"][self.layers_var[well]["gas"]["HP"]-2])) + self.biases_var[well]["gas"]["HP"][self.layers_var[well]["gas"]["HP"]-2][0] if outputs_gas[0, well, "HP"].x>0 else 0)
#            tot_oil = sum(oil_mean)
#            tot_gas = sum(gas_mean)+sum([gas_var[w]*self.s_draw.loc[s][self.wellnames[w]] for w in range(len(self.wellnames)) for s in range(self.scenarios)])/self.scenarios
#            change = [abs(changes[w, "HP", 0].x) for w in self.wellnames]
#            rowlist = [tot_exp_cap, well_cap, tot_oil, tot_gas]+chokes+gas_mean+oil_mean+gas_var+change
#        return rowlist
#
