# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 15:08:13 2018

@author: bendi
"""

# -*- coding: utf-8 -*-

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
class NN:
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

    

#    w_max_glifts = {"A2":{"HP":124200.2899}, "A3":{"HP":99956.56739}, "A5":{"HP":125615.4024}, "A6":{"HP":150090.517}, "A7":{"HP":95499.28792}, "A8":{"HP":94387.68607}, "B1":{"HP":118244.94, "LP":118244.94}, 
#               "B2":{"HP":112660.5625, "LP":112660.5625}, "B3":{"HP":238606.6016, "LP":138606.6016},
#               "B4":{"HP":90000.0709, "LP":90000.0709}, "B5":{"HP":210086.0959, "LP":210086.0959}, "B6":{"HP":117491.1591, "LP":117491.1591}, "B7":{"HP":113035.4286, "LP":113035.4286}, 
#               "C1":{"LP":106860.5264}, "C2":{"LP":132718.54}, "C3" : {"LP":98934.12}, "C4":{"LP":124718.303}}

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
#                        print(well, separator)
                        multidims[well][phase][separator], weights[well][phase][separator], biases[well][phase][separator] = t.load_2(well, phase, separator, case, net_type)
                        layers[well][phase][separator] = len(multidims[well][phase][separator])
                        if net_type=="mean" and multidims[well][phase][separator][layers[well][phase][separator]-1] > 1:
#                            print(multidims)
#                            print(layers[well][phase][separator])
                            multidims[well][phase][separator][layers[well][phase][separator]-1] -=1
                    else:
                        layers[well][phase][separator], multidims[well][phase][separator], weights[well][phase][separator], biases[well][phase][separator] = t.train(well, phase, separator, case)
        return layers, multidims, weights, biases
    
    
    # =============================================================================
    # Function to run with all wells in the problem.
    # Specify alpha to control which part of the pareto front to generate.
    # Not sure how to best handle prod_optimal... Load from file?    
    # =============================================================================
    def run_all(self, case=2, load_M = False, prod_optimal=100, alpha=1.0, w_relative_change=None, init_name=None, max_changes=15, save=False, verbose=0):
        if(case==2):
            self.wellnames = t.wellnames_2
            self.well_to_sep = t.well_to_sep_2
        else:
            self.wellnames = t.wellnames
            self.well_to_sep = t.well_to_sep
            self.platforms= t.platforms
            self.p_dict = t.p_dict
            self.p_sep_names = t.p_sep_names
            
        self.results_file = "results/mop/res_"+init_name+str(max_changes)+"_changes.csv"
        self.alpha=alpha
        try:
            res_df = pd.read_csv(self.results_file, delimiter=';')
            self.oil_optimal = res_df["tot_oil"].max()
            self.header=False
        except Exception as e:
            self.oil_optimal = prod_optimal
            self.header=True
        self.phasenames = t.phasenames
        print("MOP optimization. alpha =", self.alpha)
        print("oil_optimal:", self.oil_optimal)
        
        #Case relevant numerics
        if case==2:
            if(not w_relative_change):
                self.w_relative_change = {well : [0.4] for well in self.wellnames}
            else:
                self.w_relative_change = w_relative_change
            
            if not init_name:
                self.w_initial_prod = {well : 0 for well in self.wellnames}
                self.w_initial_vars = {well : 0 for well in self.wellnames}
            elif not isinstance(init_name, (dict)):
                #load init from file
                self.w_initial_df = t.get_robust_solution(init_name=init_name)
                self.w_initial_vars = {w:self.w_initial_df[w+"_choke"].values[0] for w in self.wellnames}
#                print(self.w_initial_vars)
                
                self.w_initial_prod = {well : 1. if self.w_initial_vars[well]>0 else 0. for well in self.wellnames}
#                print(self.w_initial_prod)
            #constraints for case 2
            else:
                #we were given a dict of initial values
#                w_initial_vars=init_name
                self.w_initial_vars={}
                for w in self.wellnames:
                    self.w_initial_vars[w] = init_name[w+"_choke"]
#                    del w_initial_vars[w+"_choke"]
                
#                print("optimization initial chokes:", w_initial_vars)
                self.w_initial_prod = {well : 1 if self.w_initial_vars[well]>0 else 0 for well in self.wellnames}
            self.tot_exp_cap = 250000
            self.well_cap = {w:54166 for w in self.wellnames}
        else:
            raise ValueError("Case 1 not implemented yet.")

            
        # =============================================================================
        # initialize an optimization model
        # =============================================================================

        self.m = Model("")
        
        #load mean and variance networks
        self.layers, self.multidims, self.weights, self.biases = self.getNeuralNets(self.LOAD, case, net_type="mean")
        self.layers_var, self.multidims_var, self.weights_var, self.biases_var = self.getNeuralNets(self.LOAD, case, net_type="var")

        # =============================================================================
        # load big M values from file
        # =============================================================================
        if(load_M):
            big_M = {well:{} for well in self.wellnames}
            for well in self.wellnames:
                for phase in self.phasenames:
                    big_M[well][phase] = {}
                    for sep in self.well_to_sep[well]:
                        big_M[well][phase][sep] = t.get_big_M(well, phase, sep)
            
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
        input_upper = {(well, sep, dim) : w_max_lims[dim][well][sep]  for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep][0])}
        input_lower = {(well, sep, dim) : w_min_lims[dim][well][sep]  for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep][0])}
        
        
        # =============================================================================
        # variable creation                    
        # =============================================================================
        inputs = self.m.addVars(input_upper.keys(), ub = input_upper, lb=input_lower, name="input", vtype=GRB.CONTINUOUS) #SEMICONT
        lambdas = self.m.addVars([(well,phase,sep, layer, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers[well][phase][sep]) for neuron in range(self.multidims[well][phase][sep][layer])], vtype = GRB.BINARY, name="lambda")
        mus = self.m.addVars([(well,phase,sep,layer, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers[well][phase][sep]) for neuron in range(self.multidims[well][phase][sep][layer])], vtype = GRB.CONTINUOUS, name="mu")
        rhos = self.m.addVars([(well,phase,sep,layer, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers[well][phase][sep]) for neuron in range(self.multidims[well][phase][sep][layer])], vtype = GRB.CONTINUOUS, name="rho")
        routes = self.m.addVars([(well, sep) for well in self.wellnames for sep in self.well_to_sep[well]], vtype = GRB.BINARY, name="routing")
#        changes = self.m.addVars([(well, sep, dim) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep])], vtype=GRB.BINARY, name="changes")
        changes = self.m.addVars([(well, sep, dim) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep][0])], vtype=GRB.BINARY, name="changes")

        #new variables to control routing decision and input/output
        outputs = self.m.addVars([(well, phase, sep) for well in self.wellnames for phase in self.phasenames for sep in self.well_to_sep[well]], vtype = GRB.CONTINUOUS, name="outputs")
        if(case==1):
            #used to subtract gas lift from gas exports
            input_dummies = self.m.addVars([(well, sep, dim) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep])], vtype = GRB.CONTINUOUS, name="input_dummies")


        #variance networks
        lambdas_var = self.m.addVars([(well,phase,sep, layer, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(self.layers_var[well][phase][sep]) for neuron in range(self.multidims_var[well][phase][sep][layer])], vtype = GRB.BINARY, name="lambda_var")
        mus_var = self.m.addVars([(well,phase,sep,layer, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(self.layers_var[well][phase][sep]) for neuron in range(self.multidims_var[well][phase][sep][layer])], vtype = GRB.CONTINUOUS, name="mu_var")
        rhos_var = self.m.addVars([(well,phase,sep,layer, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(self.layers_var[well][phase][sep]) for neuron in range(self.multidims_var[well][phase][sep][layer])], vtype = GRB.CONTINUOUS, name="rho_var")
        outputs_var = self.m.addVars([(well, phase, sep) for well in self.wellnames for phase in self.phasenames for sep in self.well_to_sep[well]], vtype = GRB.CONTINUOUS, name="outputs")



        # =============================================================================
        # NN MILP constraints creation. Mean and variance networks
        # Variables mu and rho are indexed from 1 and up since the input layer
        # is layer 0 in multidims. 
        # =============================================================================
        # mean
        self.m.addConstrs(mus[well, phase, sep, 1, neuron] - rhos[well, phase, sep, 1, neuron] - quicksum(self.weights[well][phase][sep][0][dim][neuron]*inputs[well, sep, dim] for dim in range(self.multidims[well][phase][sep][0])) == self.biases[well][phase][sep][0][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(self.multidims[well][phase][sep][1]) )
        self.m.addConstrs(mus[well, phase, sep, layer, neuron] - rhos[well, phase, sep, layer, neuron] - quicksum((self.weights[well][phase][sep][layer-1][dim][neuron]*mus[well, phase, sep, layer-1, dim]) for dim in range(self.multidims[well][phase][sep][layer-1])) == self.biases[well][phase][sep][layer-1][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(2, self.layers[well][phase][sep]) for neuron in range(self.multidims[well][phase][sep][layer]) )

        # var
        self.m.addConstrs(mus_var[well, phase, sep, 1, neuron] - rhos_var[well, phase, sep, 1, neuron] - quicksum(self.weights_var[well][phase][sep][0][dim][neuron]*inputs[well, sep, dim] for dim in range(self.multidims_var[well][phase][sep][0])) == self.biases_var[well][phase][sep][0][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(self.multidims_var[well][phase][sep][1]) )
        self.m.addConstrs(mus_var[well, phase, sep, layer, neuron] - rhos_var[well, phase, sep, layer, neuron] - quicksum((self.weights_var[well][phase][sep][layer-1][dim][neuron]*mus_var[well, phase, sep, layer-1, dim]) for dim in range(self.multidims_var[well][phase][sep][layer-1])) == self.biases_var[well][phase][sep][layer-1][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(2, self.layers_var[well][phase][sep]) for neuron in range(self.multidims_var[well][phase][sep][layer]) )
        
        #in case above do not work, manual for loops
#        for well in self.wellnames:
#            for phase in self.phasenames:
#                for sep in self.well_to_sep[well]:
#                    #mean
#                    for layer in range(2, self.layers[well][phase][sep]):
#                        self.m.addConstrs(mus[well, phase, sep, layer, neuron] - rhos[well, phase, sep, layer, neuron] - quicksum((self.weights[well][phase][sep][0][dim][neuron]*mus[well, sep, layer-1]) for dim in range(self.multidims[well][phase][sep])) == self.biases[well][phase][sep][layer][neuron] for neuron in range(len(self.biases[well][phase][sep][0])) )
#                    #var
#                    for layer in range(2, self.layers_var[well][phase][sep]):
#                        self.m.addConstrs(mus_var[well, phase, sep, layer, neuron] - rhos_var[well, phase, sep, layer, neuron] - quicksum((self.weights_var[well][phase][sep][0][dim][neuron]*mus[well, sep, layer-1]) for dim in range(self.multidims_var[well][phase][sep])) == self.biases_var[well][phase][sep][layer][neuron] for neuron in range(len(self.biases_var[well][phase][sep][0])) )

        

        #indicator constraints
        if(load_M):
            self.m.addConstrs(mus[well, phase, sep, layer, neuron] <= (big_M[well][phase][sep][neuron][0])*(1-lambdas[well, phase, sep, layer, neuron]) for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases[well][phase][sep][0])))
            self.m.addConstrs(rhos[well, phase, sep, layer, neuron] <= (big_M[well][phase][sep][neuron][1])*(lambdas[well, phase, sep, layer, neuron]) for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases[well][phase][sep][0])))
        else:
            #mean
            self.m.addConstrs( (lambdas[well, phase, sep, layer, neuron] == 1) >> (mus[well, phase, sep, layer, neuron] <= 0)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers[well][phase][sep]) for neuron in range(self.multidims[well][phase][sep][layer]))
            self.m.addConstrs( (lambdas[well, phase, sep, layer, neuron] == 0) >> (rhos[well, phase, sep, layer, neuron] <= 0)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1,self.layers[well][phase][sep]) for neuron in range(self.multidims[well][phase][sep][layer]))

            #variance
            self.m.addConstrs( (lambdas_var[well, phase, sep, layer, neuron] == 1) >> (mus_var[well, phase, sep, layer, neuron] <= 0)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_var[well][phase][sep]) for neuron in range(self.multidims_var[well][phase][sep][layer]))
            self.m.addConstrs( (lambdas_var[well, phase, sep, layer, neuron] == 0) >> (rhos_var[well, phase, sep, layer, neuron] <= 0)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_var[well][phase][sep]) for neuron in range(self.multidims_var[well][phase][sep][layer]))

        # =============================================================================
        # big-M constraints on output/routing        
        # since we model output neurons as ReLU, these are simple. may change later so that output are purely linear
        # =============================================================================
        #mean
        self.m.addConstrs( (routes[well, sep] == 1) >> (outputs[well, phase, sep] == mus[well, phase, sep, self.layers[well][phase][sep]-1, neuron])  for well in self.wellnames for phase in self.phasenames for sep in self.well_to_sep[well] for neuron in range(self.multidims[well][phase][sep][-1]))
        self.m.addConstrs( (routes[well, sep] == 0) >> (outputs[well, phase, sep] == 0) for well in self.wellnames for phase in self.phasenames for sep in self.well_to_sep[well] )
    
        #use these if model as linear
#        self.m.addConstrs( (routes[well, sep] == 1) >> (outputs[well, phase, sep] == quicksum(mus[well, phase, sep, self.layers[well][phase][sep]-1, neuron]*self.weights[well][phase][sep][self.layers[well][phase][sep]-1][neuron] for neuron in range(self.multidims[well][phase][sep][self.layers[well][phase][sep]-1])) + self.biases[well][phase][sep][self.layers[well][phase][sep]-1][0])  for well in self.wellnames for phase in self.phasenames for sep in self.well_to_sep[well])
#        self.m.addConstrs( (routes[well, sep] == 0) >> (outputs[well, phase, sep] == 0) for well in self.wellnames for phase in self.phasenames for sep in self.well_to_sep[well] )
    
        #var
        self.m.addConstrs( (routes[well, sep] == 1) >> (outputs_var[well, phase, sep] == mus_var[well, phase, sep, self.layers_var[well][phase][sep]-1, neuron])  for well in self.wellnames for phase in self.phasenames for sep in self.well_to_sep[well] for neuron in range(self.multidims_var[well][phase][sep][-1]))
        self.m.addConstrs( (routes[well, sep] == 0) >> (outputs_var[well, phase, sep] == 0) for well in self.wellnames for phase in self.phasenames for sep in self.well_to_sep[well] )
    
        #use these if model as linear
#        self.m.addConstrs( (routes[well, sep] == 1) >> (outputs_var[well, phase, sep] == quicksum(mus_var[well, phase, sep, self.layers_var[well][phase][sep]-1, neuron]*self.weights_var[well][phase][sep][self.layers_var[well][phase][sep]-1][neuron] for neuron in range(self.multidims_var[well][phase][sep][self.layers_var[well][phase][sep]-1])) + self.biases_var[well][phase][sep][self.layers_var[well][phase][sep]-1][0])  for well in self.wellnames for phase in self.phasenames for sep in self.well_to_sep[well])
#        self.m.addConstrs( (routes[well, sep] == 0) >> (outputs_var[well, phase, sep] == 0) for well in self.wellnames for phase in self.phasenames for sep in self.well_to_sep[well] )

    

    
        # =============================================================================
        # separator gas constraints
        # =============================================================================
        #single gas constraint per well in case2
        gas_constr = self.m.addConstrs(outputs[well, "gas", "HP"] <= self.well_cap[well] for well in self.wellnames)
        
    
        # =============================================================================
        # total gas export
        # =============================================================================
        if(case==1):
            exp_constr = self.m.addConstr(quicksum(outputs[well, "gas", sep] for well in self.wellnames for sep in self.well_to_sep[well]) - quicksum(input_dummies[c_well, "LP", 0] for c_well in p_dict["C"]) <= tot_exp_cap)
#        exp_constr = self.m.addConstr(quicksum(outputs[well, "gas", sep] for well in self.wellnames for sep in self.well_to_sep[well]) - quicksum(inputs[c_well, "LP", 0] for c_well in p_dict["C"]) <= tot_exp_cap)
        else:
            exp_constr = self.m.addConstr(quicksum(outputs[well, "gas", sep] for well in self.wellnames for sep in self.well_to_sep[well]) <= self.tot_exp_cap)
        
        # =============================================================================
        # routing
        # =============================================================================
        self.m.addConstrs(quicksum(routes[well, sep] for sep in self.well_to_sep[well]) <= 1 for well in self.wellnames)

        if self.alpha != "inf":
            # =============================================================================
            # alpha constraint for MOP
            # =============================================================================
            self.m.addConstr(quicksum(outputs[well, "oil", sep] for well in self.wellnames for sep in self.well_to_sep[well]) >= self.oil_optimal*self.alpha)
        
        # =============================================================================
        # change tracking and total changes
        # =============================================================================
        self.m.addConstrs(self.w_initial_vars[well] - inputs[well, sep, dim] <= changes[well, sep, dim]*self.w_initial_vars[well]*self.w_relative_change[well][dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep][0]))
        self.m.addConstrs(inputs[well, sep, dim] - self.w_initial_vars[well] <= changes[well, sep, dim]*self.w_initial_vars[well]*self.w_relative_change[well][dim]+(1-self.w_initial_prod[well])*w_max_lims[dim][well][sep]*changes[well, sep, dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep][0]))
        self.m.addConstr(quicksum(changes[well, sep, dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep][0])) <= max_changes)
        self.m.addConstrs( (self.routes[well, sep] == 0) >> (self.inputs[well, sep, dim] == 0) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1))



        # =============================================================================
        # Solver parameters
        # =============================================================================
#        self.m.setParam(GRB.Param.NumericFocus, 2)
        self.m.setParam(GRB.Param.LogToConsole, verbose)
#        self.m.setParam(GRB.Param.Heuristics, 0)
#        self.m.setParam(GRB.Param.Presolve, 0)
#        self.m.Params.timeLimit = 360.0
#        self.m.setParam(GRB.Param.LogFile, "log.txt")
        self.m.setParam(GRB.Param.DisplayInterval, 15.0)
        
        if(self.alpha == "inf"):
            #maximization of mean oil
            self.m.setObjective(quicksum(outputs[well, "oil", sep] for well in self.wellnames for sep in self.well_to_sep[well]), GRB.MAXIMIZE)
        else:
            #minimization of var gas
            self.m.setObjective(quicksum(outputs_var[well, "gas", sep] for well in self.wellnames for sep in self.well_to_sep[well]), GRB.MINIMIZE)
        

        self.m.optimize()
        
        
        if(save and case==2):
            #note, only works for case 2 as of now
#            try:
#                df = pd.read_csv(self.results_file, index_col=None)
#            except:
#                load failed, need to create new df
            df = pd.DataFrame(columns=t.MOP_res_columns)
            chokes = [inputs[well, "HP", 0].x for well in self.wellnames]
            gas_mean = [outputs[well, "gas", "HP"].x for well in self.wellnames]
            oil_mean = [outputs[well, "oil", "HP"].x for well in self.wellnames]
            gas_var = [outputs_var[well, "gas", "HP"].x for well in self.wellnames]
            oil_var = [outputs_var[well, "oil", "HP"].x for well in self.wellnames]
#            tot_oil = self.m.ObjVal
            tot_oil = sum(oil_mean)
            tot_gas = sum(gas_mean)
            rowlist = [self.alpha, tot_oil, tot_gas]+chokes+gas_mean+oil_mean+oil_var+gas_var
#            df = df.append(rowlist, ignore_index=True, axis=0)
#            df = pd.concat([df, pd.Series(rowlist)], axis=0)
            df.loc[df.shape[0]] = rowlist
#            print(rowlist)
#            newrow = pd.DataFrame(rowlist, columns=t.MOP_res_columns)
#            df.append(newrow)
            with open(self.results_file, 'a') as f:
                df.to_csv(f, sep=';', index=False, header=self.header)
        
#        for p in self.platforms:
#            print("Platform", p)
#            print("well\t", "sep\t\t", "gas\t\t\t", "oil\t\t\tgas lift\t\tchoke")
#            for well in self.p_dict[p]:
#                for sep in self.well_to_sep[well]:
#                    if(routes[well, sep].x > 0.1):
##                        print(well,"\t", sep,"\t", outputs[well, "oil", sep].x,"\t", outputs[well, "gas", sep].x, "\t", inputs[well, sep, 0].x)
#                        print(well,"\t", sep, "\t\t {0:8.2f} \t\t {1:8.2f} \t\t {2:8.2f} \t\t{3:4.4}".format(outputs[well, "gas", sep].x, outputs[well, "oil", sep].x,
#                              inputs[well, sep, 0].x, (" N/A" if self.multidims[well][phase][sep] < 2 else inputs[well, sep, 1].x)))
#            print("\n\n")
#        print("CONSTRAINTS")
#        print("gas lift slack A, B:\t",glift_constr.slack)
#        print("gas export slack:\t", exp_constr.slack)
#        print("LP gas slack:\t\t", lp_constr.slack)
#        print("HP gas slack:\t\t", hp_constr.slack)
        
    def nn(self, well, sep, load_M = False):
        self.wellnames = [well]
        self.well_to_sep[well]= [sep]
        self.platforms= [well[0]]
        self.p_dict[well[0]] = [well]
        self.p_sep_names[self.platforms[0]] = [sep]
        self.phasenames = ["oil", "gas"]
        self.run(load_M=load_M)