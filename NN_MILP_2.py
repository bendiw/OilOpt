# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:33:12 2018

@author: bendiw
"""
from gurobipy import *
import numpy as np
import tens_relu
import math
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
    def getNeuralNet(self, mode):
        weights = {well : {} for well in self.wellnames}
        biases = {well : {} for well in self.wellnames}
        multidims = {well : {} for well in self.wellnames}
        for well in self.wellnames:
            for phase in self.phasenames:
                weights[well][phase] = {}
                biases[well][phase] = {}
                multidims[well][phase] = {}
                for separator in self.well_to_sep[well]:
                    if mode==self.LOAD:
#                        print(well, separator)
                        multidims[well][phase][separator], weights[well][phase][separator], biases[well][phase][separator] = t.load(well, phase, separator)
                    else:
                        multidims[well][phase][separator], weights[well][phase][separator], biases[well][phase][separator] = t.train(well, phase, separator)
        return multidims, weights, biases
    
    
    def run_all(self, load_M = False):
        self.wellnames = t.wellnames
        self.well_to_sep = t.well_to_sep
        self.platforms= t.platforms
        self.p_dict = t.p_dict
        self.p_sep_names = t.p_sep_names
        self.phasenames = t.phasenames
        self.run(load_M=load_M)
            
    def run(self, load_M):
        
        #Case relevant numerics
        sep_p_route = {"LP": ["B", "C"], "HP":["A", "B"]}
        p_dict = {"A" : ["A2", "A3", "A5", "A6", "A7", "A8"], "B":["B1", "B2", 
             "B3", "B4", "B5", "B6", "B7"], "C":["C1", "C2", "C3", "C4"]}
    
        w_relative_change = {well : [1.0, 1.0] for well in self.wellnames}
        
        #dict with binary var describing whether or not wells are producing in initial setting
        w_initial_prod = {well : 0 for well in self.wellnames}
        
        #dict with initial values for choke, gas lift per well, {well: [gas lift, choke]}
        w_initial_vars = {well : [0,0] for well in self.wellnames}
        
        glift_caps = [675000.0]
        tot_exp_cap = 1200000
        sep_cap = {"LP": 740000, "HP":math.inf}
        glift_groups = ["A", "B"]
        max_changes = 15
        # =============================================================================
        # initialize an optimization model
        # =============================================================================
        self.m = Model("Ekofisk")
        self.multidims, self.weights, self.biases = self.getNeuralNet(self.LOAD)
        
        # =============================================================================
        # load big M values from file
        # =============================================================================
        big_M = {well:{} for well in self.wellnames}
        for well in self.wellnames:
            for phase in self.phasenames:
                big_M[well][phase] = {}
                for sep in self.well_to_sep[well]:
                    big_M[well][phase][sep] = t.get_big_M(well, phase, sep)
            
        # =============================================================================
        # get input variable bounds        
        # =============================================================================
        w_min_glift, w_max_glift = t.get_limits("gaslift_rate", self.wellnames, self.well_to_sep)
        w_min_choke, w_max_choke = t.get_limits("choke", self.wellnames, self.well_to_sep)
        w_max_lims = [w_max_glift, w_max_choke] 
        w_min_lims = [w_min_glift, w_min_choke]
        input_upper = {(well, sep, dim) : w_max_lims[dim][well][sep]  for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep])}
        input_lower = {(well, sep, dim) : w_min_lims[dim][well][sep]  for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep])}
        
        
        # =============================================================================
        # variable creation                    
        # =============================================================================
        inputs = self.m.addVars(input_upper.keys(), ub = input_upper, lb=input_lower, name="input", vtype=GRB.CONTINUOUS) #SEMICONT
        lambdas = self.m.addVars([(well,phase,sep, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases[well][phase][sep][0]))], vtype = GRB.BINARY, name="lambda")
        mus = self.m.addVars([(well,phase,sep, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases[well][phase][sep][0]))], vtype = GRB.CONTINUOUS, name="mu")
        rhos = self.m.addVars([(well,phase,sep, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases[well][phase][sep][0]))], vtype = GRB.CONTINUOUS, name="rho")
        routes = self.m.addVars([(well, sep) for well in self.wellnames for sep in self.well_to_sep[well]], vtype = GRB.BINARY, name="routing")
        changes = self.m.addVars([(well, sep, dim) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep])], vtype=GRB.BINARY, name="changes")



        #new variables to control routing decision and input/output
        outputs = self.m.addVars([(well, phase, sep) for well in self.wellnames for phase in self.phasenames for sep in self.well_to_sep[well]], vtype = GRB.CONTINUOUS, name="outputs")
        input_dummies = self.m.addVars([(well, sep, dim) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep])], vtype = GRB.CONTINUOUS, name="input_dummies")


        # =============================================================================
        # NN MILP constraints creation
        # =============================================================================
        self.m.addConstrs(mus[well, phase, sep, neuron] - rhos[well, phase, sep, neuron] - quicksum((self.weights[well][phase][sep][0][dim][neuron]*inputs[well, sep, dim]) for dim in range(self.multidims[well][phase][sep])) == self.biases[well][phase][sep][0][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases[well][phase][sep][0])) )

        #indicator constraints
        if(load_M):
            self.m.addConstrs(mus[well, phase, sep, neuron] <= (big_M[well][phase][sep][neuron][0])*(1-lambdas[well, phase, sep, neuron]) for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases[well][phase][sep][0])))
            self.m.addConstrs(rhos[well, phase, sep, neuron] <= (big_M[well][phase][sep][neuron][1])*(lambdas[well, phase, sep, neuron]) for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases[well][phase][sep][0])))

        else:
            self.m.addConstrs( (lambdas[well, phase, sep, neuron] == 1) >> (mus[well, phase, sep, neuron] <= 0)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases[well][phase][sep][0])))
            self.m.addConstrs( (lambdas[well, phase, sep, neuron] == 0) >> (rhos[well, phase, sep, neuron] <= 0)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases[well][phase][sep][0])))


        # =============================================================================
        # big-M constraints on output/routing        
        # =============================================================================
        self.m.addConstrs( (routes[well, sep] == 1) >> (outputs[well, phase, sep] == quicksum(mus[well, phase, sep, neuron]*self.weights[well][phase][sep][1][neuron] for neuron in range(len(self.biases[well]["oil"][sep][0]))) + self.biases[well][phase][sep][1][0])  for well in self.wellnames for phase in self.phasenames for sep in self.well_to_sep[well])
        self.m.addConstrs( (routes[well, sep] == 0) >> (outputs[well, phase, sep] == 0) for well in self.wellnames for phase in self.phasenames for sep in self.well_to_sep[well] )
    
        # =============================================================================
        # big-M constraints on input/routing        
        # =============================================================================
        self.m.addConstrs( (routes[well, sep] == 1) >> (input_dummies[well, sep, dim] == inputs[well, sep, dim]) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep]))
        self.m.addConstrs( (routes[well, sep] == 0) >> (input_dummies[well, sep, dim] == 0) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep]))
#        self.m.addConstrs( (routes[well, sep] == 1) >> (inputs[well, sep, dim] >= 1) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep]) )
#        self.m.addConstrs( (routes[well, sep] == 0) >> (inputs[well, sep, dim] == 0) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep]) )

    
        # =============================================================================
        # separator gas constraints
        # =============================================================================
        lp_constr = self.m.addConstr(quicksum(outputs[well, "gas", "LP"] for p in sep_p_route["LP"] for well in p_dict[p]) - quicksum(input_dummies[c_well, "LP", 0] for c_well in p_dict["C"]) <= sep_cap["LP"])

#        lp_constr = self.m.addConstr(quicksum(outputs[well, "gas", "LP"] for p in sep_p_route["LP"] for well in p_dict[p]) - quicksum(inputs[c_well, "LP", 0] for c_well in p_dict["C"]) <= sep_cap["LP"])
        hp_constr = self.m.addConstr(quicksum(outputs[well, "gas", "HP"] for p in sep_p_route["HP"] for well in p_dict[p]) <= sep_cap["HP"])
    
        # =============================================================================
        # gas lift constraints
        # =============================================================================
        glift_constr = self.m.addConstr(quicksum(input_dummies[well, sep, 0] for pform in glift_groups for well in p_dict[pform] for sep in self.well_to_sep[well]) <= glift_caps[0])
#        glift_constr = self.m.addConstr(quicksum(inputs[well, sep, 0] for pform in glift_groups for well in p_dict[pform] for sep in self.well_to_sep[well]) <= glift_caps[0])

    
        # =============================================================================
        # total gas export
        # =============================================================================
        exp_constr = self.m.addConstr(quicksum(outputs[well, "gas", sep] for well in self.wellnames for sep in self.well_to_sep[well]) - quicksum(input_dummies[c_well, "LP", 0] for c_well in p_dict["C"]) <= tot_exp_cap)
#        exp_constr = self.m.addConstr(quicksum(outputs[well, "gas", sep] for well in self.wellnames for sep in self.well_to_sep[well]) - quicksum(inputs[c_well, "LP", 0] for c_well in p_dict["C"]) <= tot_exp_cap)

        
        # =============================================================================
        # routing
        # =============================================================================
        self.m.addConstrs(quicksum(routes[well, sep] for sep in self.well_to_sep[well]) <= 1 for well in self.wellnames)

    
        # =============================================================================
        # change tracking and total changes
        # =============================================================================
        self.m.addConstrs(w_initial_vars[well][dim] - input_dummies[well, sep, dim] <= changes[well, sep, dim]*w_initial_vars[well][dim]*w_relative_change[well][dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep]))
#        self.m.addConstrs(w_initial_vars[well][dim] - inputs[well, sep, dim] <= changes[well, sep, dim]*w_initial_vars[well][dim]*w_relative_change[well][dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep]))
        
        #troublesome constraint
        self.m.addConstrs(input_dummies[well, sep, dim] - w_initial_vars[well][dim] <= changes[well, sep, dim]*w_initial_vars[well][dim]*w_relative_change[well][dim]+(1-w_initial_prod[well])*w_max_lims[dim][well][sep]*changes[well, sep, dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep]))
#        self.m.addConstrs(inputs[well, sep, dim] - w_initial_vars[well][dim] <= changes[well, sep, dim]*w_initial_vars[well][dim]*w_relative_change[well][dim]+(1-w_initial_prod[well])*w_max_lims[dim][well][sep]*changes[well, sep, dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep]))
        
        self.m.addConstr(quicksum(changes[well, sep, dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep])) <= max_changes)

    
        # =============================================================================
        # objective
        # =============================================================================
#        self.m.setParam(GRB.Param.NumericFocus, 2)
#        self.m.setParam(GRB.Param.LogToConsole, 0)
#        self.m.setObjective(quicksum(mus[well, "oil", sep, neuron]*self.weights[well]["oil"][sep][1][neuron] for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases[well]["oil"][sep][0]))) + quicksum([self.biases[well]["oil"][sep][1][0] for well in self.wellnames for sep in self.well_to_sep[well]]), GRB.MAXIMIZE)
        self.m.setObjective(quicksum(outputs[well, "oil", sep] for well in self.wellnames for sep in self.well_to_sep[well]), GRB.MAXIMIZE)
#        self.m.setParam(GRB.Param.Heuristics, 0)
#        self.m.setParam(GRB.Param.LogFile, "log.txt")
        self.m.setParam(GRB.Param.DisplayInterval, 15.0)
#        self.m.setParam(GRB.Param.Presolve, 0)
#        self.m.Params.timeLimit = 360.0

        self.m.optimize()
        
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