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
                        layers[well][phase][separator], multidims[well][phase][separator], weights[well][phase][separator], biases[well][phase][separator] = t.load(well, phase, separator, case, net_type)
                    else:
                        layers[well][phase][separator], multidims[well][phase][separator], weights[well][phase][separator], biases[well][phase][separator] = t.train(well, phase, separator, case)
        return layers, multidims, weights, biases
    
    
    # =============================================================================
    # Function to run with all wells in the problem.
    # Specify alpha to control which part of the pareto front to generate.
    # Not sure how to best handle prod_optimal... Load from file?    
    # =============================================================================
    def run_all(self, case=2, load_M = False, prod_optimal=100, alpha=1.0):
        if(case==2):
            self.wellnames = t.wellnames_2
            self.well_to_sep = t.well_to_sep_2
        else:
            self.wellnames = t.wellnames
            self.well_to_sep = t.well_to_sep
            self.platforms= t.platforms
            self.p_dict = t.p_dict
            self.p_sep_names = t.p_sep_names
            
        self.results_file = "results/mop/res.csv"
        self.alpha=alpha
        self.prod_optimal = prod_optimal
        self.phasenames = t.phasenames
        self.run(load_M=load_M, case=case)
            
    def run(self, load_M, case=2, save=True):
        
        #Case relevant numerics
        if case==1:
            sep_p_route = {"LP": ["B", "C"], "HP":["A", "B"]}
            p_dict = {"A" : ["A2", "A3", "A5", "A6", "A7", "A8"], "B":["B1", "B2", 
                 "B3", "B4", "B5", "B6", "B7"], "C":["C1", "C2", "C3", "C4"]}
        
            #probably not needed for MOP
#            max_changes = 15 
#            w_relative_change = {well : [1.0, 1.0] for well in self.wellnames}
            #dict with binary var describing whether or not wells are producing in initial setting
#            w_initial_prod = {well : 0 for well in self.wellnames}
            #dict with initial values for choke, gas lift per well, {well: [gas lift, choke]}
#            w_initial_vars = {well : [0,0] for well in self.wellnames}
            
            glift_caps = [675000.0]
            tot_exp_cap = 1200000
            sep_cap = {"LP": 740000, "HP":math.inf}
            glift_groups = ["A", "B"]
        else:
            
            #probably not needed for MOP
#            w_relative_change = {well : [1.0, 1.0] for well in self.wellnames}
            #dict with binary var describing whether or not wells are producing in initial setting
#            w_initial_prod = {well : 0 for well in self.wellnames}
            #dict with initial values for choke, gas lift per well, {well: [gas lift, choke]}
#            w_initial_vars = {well : [0,0] for well in self.wellnames}
            
            tot_exp_cap = 6000000
            well_cap = 1300000

            
        # =============================================================================
        # initialize an optimization model
        # =============================================================================
        self.m = Model("Ekofisk")
        
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
        
        w_max_lims = [w_max_glift, w_max_choke] 
        w_min_lims = [w_min_glift, w_min_choke]
        input_upper = {(well, sep, dim) : w_max_lims[dim][well][sep]  for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep])}
        input_lower = {(well, sep, dim) : w_min_lims[dim][well][sep]  for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep])}
        
        
        # =============================================================================
        # variable creation                    
        # =============================================================================
        inputs = self.m.addVars(input_upper.keys(), ub = input_upper, lb=input_lower, name="input", vtype=GRB.CONTINUOUS) #SEMICONT
        lambdas = self.m.addVars([(well,phase,sep, layers, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(self.layers[well][phase][sep]) for neuron in range(len(self.biases[well][phase][sep][0]))], vtype = GRB.BINARY, name="lambda")
        mus = self.m.addVars([(well,phase,sep,layers, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(self.layers[well][phase][sep]) for neuron in range(len(self.biases[well][phase][sep][0]))], vtype = GRB.CONTINUOUS, name="mu")
        rhos = self.m.addVars([(well,phase,sep,layers, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(self.layers[well][phase][sep]) for neuron in range(len(self.biases[well][phase][sep][0]))], vtype = GRB.CONTINUOUS, name="rho")
        routes = self.m.addVars([(well, sep) for well in self.wellnames for sep in self.well_to_sep[well]], vtype = GRB.BINARY, name="routing")
        changes = self.m.addVars([(well, sep, dim) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep])], vtype=GRB.BINARY, name="changes")

        #new variables to control routing decision and input/output
        outputs = self.m.addVars([(well, phase, sep) for well in self.wellnames for phase in self.phasenames for sep in self.well_to_sep[well]], vtype = GRB.CONTINUOUS, name="outputs")
        if(case==1):
            input_dummies = self.m.addVars([(well, sep, dim) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep])], vtype = GRB.CONTINUOUS, name="input_dummies")



        #variance networks
        lambdas_var = self.m.addVars([(well,phase,sep, layers, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(self.layers_var[well][phase][sep]) for neuron in range(len(self.biases_var[well][phase][sep][0]))], vtype = GRB.BINARY, name="lambda_var")
        mus_var = self.m.addVars([(well,phase,sep,layers, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(self.layers_var[well][phase][sep]) for neuron in range(len(self.biases_var[well][phase][sep][0]))], vtype = GRB.CONTINUOUS, name="mu_var")
        rhos_var = self.m.addVars([(well,phase,sep,layers, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(self.layers_var[well][phase][sep]) for neuron in range(len(self.biases_var[well][phase][sep][0]))], vtype = GRB.CONTINUOUS, name="rho_var")
        outputs_var = self.m.addVars([(well, phase, sep) for well in self.wellnames for phase in self.phasenames for sep in self.well_to_sep[well]], vtype = GRB.CONTINUOUS, name="outputs")



        # =============================================================================
        # NN MILP constraints creation. Mean and variance networks
        # =============================================================================
        # first layer
        self.m.addConstrs(mus[well, phase, sep, 0, neuron] - rhos[well, phase, sep, 0, neuron] - quicksum((self.weights[well][phase][sep][0][dim][neuron]*inputs[well, sep, dim]) for dim in range(self.multidims[well][phase][sep])) == self.biases[well][phase][sep][0][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases[well][phase][sep][0])) )
        self.m.addConstrs(mus_var[well, phase, sep, 0, neuron] - rhos_var[well, phase, sep, 0, neuron] - quicksum((self.weights_var[well][phase][sep][0][dim][neuron]*inputs[well, sep, dim]) for dim in range(self.multidims_var[well][phase][sep])) == self.biases_var[well][phase][sep][0][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases_var[well][phase][sep][0])) )

        #other layers - not sure if addconstrs works like this
        self.m.addConstrs(mus[well, phase, sep, layer, neuron] - rhos[well, phase, sep, layer, neuron] - quicksum((self.weights[well][phase][sep][0][dim][neuron]*mus[well, sep, layer-1]) for dim in range(self.multidims[well][phase][sep])) == self.biases[well][phase][sep][layer][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(self.layers[well][phase][sep]-1) for neuron in range(len(self.biases[well][phase][sep][0])) )
        self.m.addConstrs(mus_var[well, phase, sep, layer, neuron] - rhos_var[well, phase, sep, layer, neuron] - quicksum((self.weights_var[well][phase][sep][0][dim][neuron]*mus_var[well, sep, layer-1]) for dim in range(self.multidims_var[well][phase][sep])) == self.biases_var[well][phase][sep][layer][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(self.layers_var[well][phase][sep]-1) for neuron in range(len(self.biases_var[well][phase][sep][0])) )
        
        #in case above do not work, manual for loops
#        for well in self.wellnames:
#            for phase in self.phasenames:
#                for sep in self.well_to_sep[well]:
#                    #mean
#                    for layer in range(1, self.layers[well][phase][sep]):
#                        self.m.addConstrs(mus[well, phase, sep, layer, neuron] - rhos[well, phase, sep, layer, neuron] - quicksum((self.weights[well][phase][sep][0][dim][neuron]*mus[well, sep, layer-1]) for dim in range(self.multidims[well][phase][sep])) == self.biases[well][phase][sep][layer][neuron] for neuron in range(len(self.biases[well][phase][sep][0])) )
#                    #var
#                    for layer in range(1, self.layers_var[well][phase][sep]):
#                        self.m.addConstrs(mus_var[well, phase, sep, layer, neuron] - rhos_var[well, phase, sep, layer, neuron] - quicksum((self.weights_var[well][phase][sep][0][dim][neuron]*mus[well, sep, layer-1]) for dim in range(self.multidims_var[well][phase][sep])) == self.biases_var[well][phase][sep][layer][neuron] for neuron in range(len(self.biases_var[well][phase][sep][0])) )

        
#        for l in range()

        #indicator constraints
        if(load_M):
            self.m.addConstrs(mus[well, phase, sep, layer, neuron] <= (big_M[well][phase][sep][neuron][0])*(1-lambdas[well, phase, sep, neuron]) for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases[well][phase][sep][0])))
            self.m.addConstrs(rhos[well, phase, sep, layer, neuron] <= (big_M[well][phase][sep][neuron][1])*(lambdas[well, phase, sep, neuron]) for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases[well][phase][sep][0])))
        else:
            #mean
            self.m.addConstrs( (lambdas[well, phase, sep, layer, neuron] == 1) >> (mus[well, phase, sep, layer, neuron] <= 0)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(self.layers[well][phase][sep]) for neuron in range(len(self.biases[well][phase][sep][0])))
            self.m.addConstrs( (lambdas[well, phase, sep, layer, neuron] == 0) >> (rhos[well, phase, sep, layer, neuron] <= 0)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(self.layers[well][phase][sep]) for neuron in range(len(self.biases[well][phase][sep][0])))

            #variance
            self.m.addConstrs( (lambdas_var[well, phase, sep, layer, neuron] == 1) >> (mus_var[well, phase, sep, layer, neuron] <= 0)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(self.layers_var[well][phase][sep]) for neuron in range(len(self.biases_var[well][phase][sep][0])))
            self.m.addConstrs( (lambdas_var[well, phase, sep, layer, neuron] == 0) >> (rhos_var[well, phase, sep, layer, neuron] <= 0)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(self.layers_var[well][phase][sep]) for neuron in range(len(self.biases_var[well][phase][sep][0])))

        # =============================================================================
        # big-M constraints on output/routing        
        # =============================================================================
        #mean
        self.m.addConstrs( (routes[well, sep] == 1) >> (outputs[well, phase, sep] == quicksum(mus[well, phase, sep, self.layers[well][phase][sep], neuron]*self.weights[well][phase][sep][self.layers[well][phase][sep]][neuron] for neuron in range(len(self.biases[well]["oil"][sep][0]))) + self.biases[well][phase][sep][self.layers[well][phase][sep]][0])  for well in self.wellnames for phase in self.phasenames for sep in self.well_to_sep[well])
        self.m.addConstrs( (routes[well, sep] == 0) >> (outputs[well, phase, sep] == 0) for well in self.wellnames for phase in self.phasenames for sep in self.well_to_sep[well] )
    
        #var
        self.m.addConstrs( (routes[well, sep] == 1) >> (outputs_var[well, phase, sep] == quicksum(mus_var[well, phase, sep, self.layers_var[well][phase][sep], neuron]*self.weights_var[well][phase][sep][self.layers_var[well][phase][sep]][neuron] for neuron in range(len(self.biases_var[well]["oil"][sep][0]))) + self.biases_var[well][phase][sep][self.layers_var[well][phase][sep]][0])  for well in self.wellnames for phase in self.phasenames for sep in self.well_to_sep[well])
        self.m.addConstrs( (routes[well, sep] == 0) >> (outputs_var[well, phase, sep] == 0) for well in self.wellnames for phase in self.phasenames for sep in self.well_to_sep[well] )

    
        # =============================================================================
        # big-M constraints on input/routing        
        # =============================================================================
        if(case==1):
            #do not use input dummies in case 2 since these are needed for gas lift only
            self.m.addConstrs( (routes[well, sep] == 1) >> (input_dummies[well, sep, dim] == inputs[well, sep, dim]) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep]))
            self.m.addConstrs( (routes[well, sep] == 0) >> (input_dummies[well, sep, dim] == 0) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep]))
    #        self.m.addConstrs( (routes[well, sep] == 1) >> (inputs[well, sep, dim] >= 1) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep]) )
#            self.m.addConstrs( (routes[well, sep] == 0) >> (inputs[well, sep, dim] == 0) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep]) )

    
        # =============================================================================
        # separator gas constraints
        # =============================================================================
        if(case==1):
            lp_constr = self.m.addConstr(quicksum(outputs[well, "gas", "LP"] for p in sep_p_route["LP"] for well in p_dict[p]) - quicksum(input_dummies[c_well, "LP", 0] for c_well in p_dict["C"]) <= sep_cap["LP"])
    
    #        lp_constr = self.m.addConstr(quicksum(outputs[well, "gas", "LP"] for p in sep_p_route["LP"] for well in p_dict[p]) - quicksum(inputs[c_well, "LP", 0] for c_well in p_dict["C"]) <= sep_cap["LP"])
            hp_constr = self.m.addConstr(quicksum(outputs[well, "gas", "HP"] for p in sep_p_route["HP"] for well in p_dict[p]) <= sep_cap["HP"])
        else:
            gas_constr = self.m.addConstrs(outputs[well, gas, "HP"] <= well_cap for well in self.wellnames)
            
        # =============================================================================
        # gas lift constraints
        # =============================================================================
        if(case==1):
            glift_constr = self.m.addConstr(quicksum(input_dummies[well, sep, 0] for pform in glift_groups for well in p_dict[pform] for sep in self.well_to_sep[well]) <= glift_caps[0])
#        glift_constr = self.m.addConstr(quicksum(inputs[well, sep, 0] for pform in glift_groups for well in p_dict[pform] for sep in self.well_to_sep[well]) <= glift_caps[0])

    
        # =============================================================================
        # total gas export
        # =============================================================================
        if(case==1):
            exp_constr = self.m.addConstr(quicksum(outputs[well, "gas", sep] for well in self.wellnames for sep in self.well_to_sep[well]) - quicksum(input_dummies[c_well, "LP", 0] for c_well in p_dict["C"]) <= tot_exp_cap)
#        exp_constr = self.m.addConstr(quicksum(outputs[well, "gas", sep] for well in self.wellnames for sep in self.well_to_sep[well]) - quicksum(inputs[c_well, "LP", 0] for c_well in p_dict["C"]) <= tot_exp_cap)
        else:
            self.m.addConstr(quicksum(outputs[well, "gas", sep] for well in self.wellnames for sep in self.well_to_sep[well]) <= tot_exp_cap)
        
        # =============================================================================
        # routing
        # =============================================================================
        self.m.addConstrs(quicksum(routes[well, sep] for sep in self.well_to_sep[well]) <= 1 for well in self.wellnames)

    
        # =============================================================================
        # alpha constraint for MOP
        # =============================================================================
        self.m.addConstr(quicksum(outputs[well, "oil", sep] for well in self.wellnames for sep in self.well_to_sep[well]) >= self.oil_optimal*self.alpha)
        
        # =============================================================================
        # change tracking and total changes
        # Probably do not care about these when generating pareto front for now
        # =============================================================================
#        if(case==1):
#            self.m.addConstrs(w_initial_vars[well][dim] - input_dummies[well, sep, dim] <= changes[well, sep, dim]*w_initial_vars[well][dim]*w_relative_change[well][dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep]))
##            self.m.addConstrs(w_initial_vars[well][dim] - inputs[well, sep, dim] <= changes[well, sep, dim]*w_initial_vars[well][dim]*w_relative_change[well][dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep]))
#            #troublesome constraint
#            self.m.addConstrs(input_dummies[well, sep, dim] - w_initial_vars[well][dim] <= changes[well, sep, dim]*w_initial_vars[well][dim]*w_relative_change[well][dim]+(1-w_initial_prod[well])*w_max_lims[dim][well][sep]*changes[well, sep, dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep]))
#    #        self.m.addConstrs(inputs[well, sep, dim] - w_initial_vars[well][dim] <= changes[well, sep, dim]*w_initial_vars[well][dim]*w_relative_change[well][dim]+(1-w_initial_prod[well])*w_max_lims[dim][well][sep]*changes[well, sep, dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep]))
#            self.m.addConstr(quicksum(changes[well, sep, dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep])) <= max_changes)
    
        # =============================================================================
        # objective and parameters
        # =============================================================================
#        self.m.setParam(GRB.Param.NumericFocus, 2)
#        self.m.setParam(GRB.Param.LogToConsole, 0)
#        self.m.setParam(GRB.Param.Heuristics, 0)
#        self.m.setParam(GRB.Param.Presolve, 0)
#        self.m.Params.timeLimit = 360.0
#        self.m.setParam(GRB.Param.LogFile, "log.txt")
        self.m.setObjective(quicksum(outputs[well, "oil", sep] for well in self.wellnames for sep in self.well_to_sep[well]), GRB.MAXIMIZE)
#        self.m.setObjective(quicksum(outputs_var[well, "oil", sep] for well in self.wellnames for sep in self.well_to_sep[well]), GRB.MINIMIZE)
        self.m.setParam(GRB.Param.DisplayInterval, 15.0)

        self.m.optimize()
        
        if(save):
            #note, only works for case 2 as of now
            assert(case==1)
            df = pd.read_csv(self.results_file)
            chokes = [inputs[well, "gas", "HP"].x for well in self.wellnames]
            gas_mean = [outputs[well, "gas", "HP"] for well in self.wellnames]
            oil_mean = [outputs[well, "oil", "HP"] for well in self.wellnames]
            gas_var = [outputs_var[well, "gas", "HP"] for well in self.wellnames]
            oil_var = [outputs_var[well, "oil", "HP"] for well in self.wellnames]
            tot_oil = self.m.ObjVal
            newrow = pd.DataFrame(self.alpha+tot_oil+chokes+oil_mean+gas_mean+oil_var+gas_var, columns=df.columns)
        
        for p in self.platforms:
            print("Platform", p)
            print("well\t", "sep\t\t", "gas\t\t\t", "oil\t\t\tgas lift\t\tchoke")
            for well in self.p_dict[p]:
                for sep in self.well_to_sep[well]:
                    if(routes[well, sep].x > 0.1):
#                        print(well,"\t", sep,"\t", outputs[well, "oil", sep].x,"\t", outputs[well, "gas", sep].x, "\t", inputs[well, sep, 0].x)
                        print(well,"\t", sep, "\t\t {0:8.2f} \t\t {1:8.2f} \t\t {2:8.2f} \t\t{3:4.4}".format(outputs[well, "gas", sep].x, outputs[well, "oil", sep].x,
                              inputs[well, sep, 0].x, (" N/A" if self.multidims[well][phase][sep] < 2 else inputs[well, sep, 1].x)))
            print("\n\n")
        print("CONSTRAINTS")
        print("gas lift slack A, B:\t",glift_constr.slack)
        print("gas export slack:\t", exp_constr.slack)
        print("LP gas slack:\t\t", lp_constr.slack)
        print("HP gas slack:\t\t", hp_constr.slack)
        
    def nn(self, well, sep, load_M = False):
        self.wellnames = [well]
        self.well_to_sep[well]= [sep]
        self.platforms= [well[0]]
        self.p_dict[well[0]] = [well]
        self.p_sep_names[self.platforms[0]] = [sep]
        self.phasenames = ["oil", "gas"]
        self.run(load_M=load_M)