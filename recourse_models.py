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
    # Function to build a model with all wells in the problem.
    # =============================================================================
    def init(self, case=2,
                num_scen = 1000, lower=-4, upper=4, phase="gas", sep="HP", save=True,store_init=False, init_name=None, 
                max_changes=15, w_relative_change=None, stability_iter=None, distr="truncnorm", lock_wells=None, scen_const=None, recourse_iter=False):
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
        self.save=save
        self.neg_change_constr = None
        self.pos_change_constr = None
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
        self.w_max_lims = [w_max_choke, w_max_glift] 
        self.w_min_lims = [w_min_choke, w_min_glift]
        self.input_upper = {(well, sep, dim) : self.w_max_lims[dim][well][sep]  for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1)}
        self.input_lower = {(well, sep, dim) : self.w_min_lims[dim][well][sep]  for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1)}
        
        #new variables to control routing decision and input/output
        self.outputs_gas = self.m.addVars([(scenario, well, sep) for well in self.wellnames for sep in self.well_to_sep[well] for scenario in range(self.scenarios)], vtype = GRB.CONTINUOUS, name="outputs_gas")
        self.outputs_oil = self.m.addVars([(well, sep) for well in self.wellnames for sep in self.well_to_sep[well]], vtype = GRB.CONTINUOUS, name="outputs_oil")
        self.routes = self.m.addVars([(well, sep) for well in self.wellnames for sep in self.well_to_sep[well]], vtype = GRB.BINARY, name="routing")
        self.changes = self.m.addVars([(well, sep, dim) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1)], vtype=GRB.BINARY, name="changes")
        self.inputs = self.m.addVars(self.input_upper.keys(), ub = self.input_upper, lb=self.input_lower, name="input", vtype=GRB.SEMICONT) #SEMICONT
        
        
        #input forcing to track changes
        self.m.addConstrs( (self.routes[well, sep] == 0) >> (self.inputs[well, sep, dim] <= 0) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1))

        
        # =============================================================================
        # separator gas constraints
        # =============================================================================
        self.gas_constr = self.m.addConstrs(self.outputs_gas[scenario, well, "HP"] <= self.well_cap[well] for well in self.wellnames for scenario in range(self.scenarios))
        self.exp_constr = self.m.addConstrs(quicksum(self.outputs_gas[scenario, well, sep] for well in self.wellnames for sep in self.well_to_sep[well]) <= self.tot_exp_cap for scenario in range(self.scenarios))
        
        # =============================================================================
        # routing
        # =============================================================================
        self.m.addConstrs(quicksum(self.routes[well, sep] for sep in self.well_to_sep[well]) <= 1 for well in self.wellnames)
        # =============================================================================
#         change tracking and total changes
#         =============================================================================
        self.set_chokes(self.w_initial_vars)
        
        self.change_constr = self.m.addConstr(quicksum(self.changes[well, sep, dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1)) <= self.max_changes)
        self.m.setObjective( quicksum(self.outputs_oil[well, sep] for well in self.wellnames for sep in self.well_to_sep[well]), GRB.MAXIMIZE)


    def set_changes(self, max_changes):
        self.change_constr.setAttr("rhs", max_changes)
        
    def set_tot_gas(self, gas):
        for s in range(self.scenarios):
            self.exp_constr[s].setAttr("rhs", gas)
    
    
    #set indiv gas cap. if no well is specified, all wells are affected
    def set_indiv_gas(self, gas, wells=None):
        if wells:
            if(isinstance(wells, (list))):
                to_change = wells
            else:
                to_change = [wells]
        else:
            to_change = self.wellnames
        print("tocange:", to_change)
        for w in to_change:
            for s in range(self.scenarios):
                self.gas_constr[w, s].setAttr("rhs", gas)
        
    def set_chokes(self, w_initial_vars):
        self.w_initial_vars = w_initial_vars
        self.w_initial_prod = {well : 1. if self.w_initial_vars[well]>0 else 0. for well in self.wellnames}
        if(self.neg_change_constr is not None):
            self.m.remove(self.neg_change_constr)
        if(self.pos_change_constr is not None):
            self.m.remove(self.pos_change_constr)
            
        self.neg_change_constr = self.m.addConstrs(self.w_initial_vars[well] - self.inputs[well, sep, dim] <= self.changes[well, sep, dim]*self.w_initial_vars[well]*self.w_relative_change[well][dim] + 
                          (self.w_initial_vars[well]*(1-quicksum(self.routes[well, separ] for separ in self.well_to_sep[well]))*(1-self.w_relative_change[well][dim])) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1))
        
        self.pos_change_constr = self.m.addConstrs(self.inputs[well, sep, dim] - self.w_initial_vars[well] <= self.changes[well, sep, dim]*self.w_initial_vars[well]*self.w_relative_change[well][dim]+
                          (1-self.w_initial_prod[well])*self.w_max_lims[dim][well][sep]*self.changes[well, sep, dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1))

    def solve(self, verbose=1):
        # =============================================================================
        # Solver parameters
        # =============================================================================
#        self.m.setParam(GRB.Param.NumericFocus, 2)
        self.verbose = verbose
        self.m.setParam(GRB.Param.LogToConsole, self.verbose)
#        self.m.setParam(GRB.Param.Heuristics, 0)
#        self.m.setParam(GRB.Param.Presolve, 0)
#        self.m.Params.timeLimit = 360.0
#        self.m.setParam(GRB.Param.LogFile, "log.txt")
        self.m.setParam(GRB.Param.DisplayInterval, 15.0)
        #maximization of mean oil. no need to take mean over scenarios since only gas is scenario dependent
        self.m.optimize()
        
    def get_solution(self):        
        df = pd.DataFrame(columns=t.robust_res_columns_SOS2) 
        chokes = [self.inputs[well, "HP", 0].x if self.outputs_gas[0, well, "HP"].x>0 else 0 for well in self.wellnames]
        rowlist=[]
        if(self.case==2 and self.save):
            gas_mean = np.zeros(len(self.wellnames))
#            gas_var=[]
            w = 0
            for well in self.wellnames:
                for scenario in range(self.scenarios):
                    gas_mean[w] += self.outputs_gas[scenario, well, "HP"].x
#                gas_var.append(sum(self.weights_var[well]["gas"]["HP"][self.layers_var[well]["gas"]["HP"]-2][neuron][0] * (self.mus_var[well, "gas", "HP", self.layers_var[well]["gas"]["HP"]-2, neuron].x) for neuron in range(self.multidims_var[well]["gas"]["HP"][self.layers_var[well]["gas"]["HP"]-2])) + self.biases_var[well]["gas"]["HP"][self.layers_var[well]["gas"]["HP"]-2][0])

                w += 1
            gas_mean = (gas_mean/float(self.scenarios)).tolist()
            oil_mean = [self.outputs_oil[well, "HP"].x for well in self.wellnames]
#            oil_var = [outputs_]
            tot_oil = sum(oil_mean)
            tot_gas = sum(gas_mean)
            change = [abs(self.changes[w, "HP", 0].x) for w in self.wellnames]
#            print(df.columns)
            rowlist = [self.scenarios, self.tot_exp_cap]+ [tot_oil, tot_gas]+list(self.well_cap.values())+chokes+gas_mean+oil_mean+change
#            print(len(rowlist))
            if(self.store_init):
                df.rename(columns={"scenarios": "name"}, inplace=True)
                rowlist[0] = self.init_name
                df.loc[df.shape[0]] = rowlist

                head = not os.path.isfile(self.results_file_init)
                with open(self.results_file_init, 'a') as f:
                    df.to_csv(f, sep=';', index=False, header=head)
            else:
                df.loc[df.shape[0]] = rowlist
                head = not os.path.isfile(self.results_file)
                with open(self.results_file, 'a') as f:
                    df.to_csv(f, sep=';', index=False, header=head)
        elif self.recourse_iter:
            oil_mean = [self.outputs_oil[well, "HP"].x for well in self.wellnames]
            gas_mean = []
#            gas_var = []
            for well in self.wellnames:
                gas_mean.append(sum(self.weights[well]["gas"]["HP"][self.layers[well]["gas"]["HP"]-2][neuron][0] * self.mus[well, "gas", "HP", self.layers[well]["gas"]["HP"]-2, neuron].x for neuron in range(self.multidims[well]["gas"]["HP"][self.layers[well]["gas"]["HP"]-2]) ) + self.biases[well]["gas"]["HP"][self.layers[well]["gas"]["HP"]-2][0] if self.outputs_gas[0, well, "HP"].x>0 else 0)
#                gas_var.append(sum(self.weights_var[well]["gas"]["HP"][self.layers_var[well]["gas"]["HP"]-2][neuron][0] * (mus_var[well, "gas", "HP", self.layers_var[well]["gas"]["HP"]-2, neuron].x) for neuron in range(self.multidims_var[well]["gas"]["HP"][self.layers_var[well]["gas"]["HP"]-2])) + self.biases_var[well]["gas"]["HP"][self.layers_var[well]["gas"]["HP"]-2][0] if self.outputs_gas[0, well, "HP"].x>0 else 0)
            tot_oil = sum(oil_mean)
#            tot_gas = sum(gas_mean)+sum([gas_var[w]*self.s_draw.loc[s][self.wellnames[w]] for w in range(len(self.wellnames)) for s in range(self.scenarios)])/self.scenarios
            change = [abs(self.changes[w, "HP", 0].x) for w in self.wellnames]
            rowlist = [self.tot_exp_cap]+ [ tot_oil, tot_gas]+list(self.well_cap.values())+chokes+gas_mean+oil_mean+change
        return df
    
    def get_chokes(self):
        chokes = {well:self.inputs[well, "HP", 0].x if self.outputs_gas[0, well, "HP"].x>0 else 0 for well in self.wellnames}
        return chokes
    
class NN(Recourse_Model):
    # =============================================================================
    # get neural nets either by loading existing ones or training new ones
    # =============================================================================
    def getNeuralNets(self, mode, case, phase, net_type="scen"):
        if phase=="gas":
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
                            multidims[scenario][well][separator], weights[scenario][well][separator], biases[scenario][well][separator] = t.load_2(well, phase, separator, case, net_type, scenario)
                            layers[scenario][well][separator] = len(multidims[scenario][well][separator])
                            if net_type=="mean" and multidims[scenario][well][separator][ layers[scenario][well][separator]-1 ] > 1:
                                multidims[scenario][well][separator][ layers[scenario][well][separator]-1 ] -=1
                        else:
                            layers[scenario][well][separator], multidims[scenario][well][separator], weights[scenario][well][separator], biases[scenario][well][separator] = t.train(well, phase, separator, case)
        elif phase == "oil":
            weights = {well : {} for well in self.wellnames}
            biases = {well : {} for well in self.wellnames}
            multidims = {well : {} for well in self.wellnames}
            layers = {well : {} for well in self.wellnames}
            for well in self.wellnames:
                weights[well] = {}
                biases[well] = {}
                multidims[well] = {}
                layers[well] = {} 
                for separator in self.well_to_sep[well]:
                    if mode==self.LOAD:
                        multidims[well][separator], weights[well][separator], biases[well][separator] = t.load_2(well, phase, separator, case, net_type, scen=0)
                        layers[well][separator] = len(multidims[well][separator])
                        if net_type=="mean" and multidims[well][separator][layers[well][separator]-1 ] > 1:
                            multidims[well][separator][ layers[well][separator]-1 ] -=1
                    else:
                        layers[well][separator], multidims[well][separator], weights[well][separator], biases[well][separator] = t.train(well, phase, separator, case)
            
        return layers, multidims, weights, biases
    
    
    # =============================================================================
    # Function to build a model with all wells in the problem.
    # =============================================================================
    def init(self, case=2,
                num_scen = 10, lower=-4, upper=4, phase="gas", sep="HP", save=True,store_init=False, init_name=None, 
                max_changes=15, w_relative_change=None, stability_iter=None, distr="truncnorm", lock_wells=None, scen_const=None, recourse_iter=False, verbose=1):

        Recourse_Model.init(self, case, num_scen, lower, upper, phase, sep, save, store_init, init_name, max_changes, w_relative_change, stability_iter, distr, lock_wells, scen_const, recourse_iter)        
        self.results_file = "results/robust/res_NN.csv"
        #load mean and variance networks
        self.layers_gas, self.multidims_gas, self.weights_gas, self.biases_gas = self.getNeuralNets(self.LOAD, case, net_type="scen", phase="gas")
        self.layers_oil, self.multidims_oil, self.weights_oil, self.biases_oil = self.getNeuralNets(self.LOAD, case, net_type="scen", phase="oil")

        # =============================================================================
        # variable creation                    
        # Gas networks scenario dependent, oil networks are not
        # =============================================================================
        self.lambdas_gas = self.m.addVars([(scenario, well,sep, layer, neuron) for scenario in range(self.scenarios) for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_gas[scenario][well][sep]-1) for neuron in range(self.multidims_gas[scenario][well][sep][layer])], vtype = GRB.BINARY, name="lambda_gas")
        self.mus_gas = self.m.addVars([(scenario, well,sep,layer, neuron) for scenario in range(self.scenarios)  for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_gas[scenario][well][sep]-1) for neuron in range(self.multidims_gas[scenario][well][sep][layer])], vtype = GRB.CONTINUOUS, name="mu_gas")
        self.rhos_gas = self.m.addVars([(scenario, well,sep,layer, neuron) for scenario in range(self.scenarios) for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_gas[scenario][well][sep]-1) for neuron in range(self.multidims_gas[scenario][well][sep][layer])], vtype = GRB.CONTINUOUS, name="rho_gas")
        
        self.lambdas_oil = self.m.addVars([(well,sep, layer, neuron) for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_oil[well][sep]-1) for neuron in range(self.multidims_oil[well][sep][layer])], vtype = GRB.BINARY, name="lambda_oil")
        self.mus_oil = self.m.addVars([(well,sep,layer, neuron)  for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_oil[well][sep]-1) for neuron in range(self.multidims_oil[well][sep][layer])], vtype = GRB.CONTINUOUS, name="mu_oil")
        self.rhos_oil = self.m.addVars([(well,sep,layer, neuron)   for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_oil[well][sep]-1) for neuron in range(self.multidims_oil[well][sep][layer])], vtype = GRB.CONTINUOUS, name="rho_oil")

        # NN MILP constraints creation. Mean and variance networks.
        # Variables mu and rho are indexed from 1 and up since the input layer is layer 0 in multidims. 
        # =============================================================================
        # gas
        self.m.addConstrs(self.mus_gas[scenario, well, sep, 1, neuron] - self.rhos_gas[scenario, well, sep, 1, neuron] - quicksum(self.weights_gas[scenario][well][sep][0][dim][neuron]*self.inputs[well, sep, dim] for dim in range(self.multidims_gas[scenario][well][sep][0])) == self.biases_gas[scenario][well][sep][0][neuron] for scenario in range(self.scenarios) for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(self.multidims_gas[scenario][well][sep][1]))
        self.m.addConstrs(self.mus_gas[scenario, well, sep, layer, neuron] - self.rhos_gas[scenario, well, sep, layer, neuron] - quicksum((self.weights_gas[scenario][well][sep][layer-1][dim][neuron]*self.mus_gas[scenario, well, sep, layer-1, dim]) for dim in range(self.multidims_gas[scenario][well][sep][layer-1])) == self.biases_gas[scenario][well][sep][layer-1][neuron] for scenario in range(self.scenarios) for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(2, self.layers_gas[scenario][well][sep]-1) for neuron in range(self.multidims_gas[scenario][well][sep][layer]) )
        # oil
        self.m.addConstrs(self.mus_oil[well, sep, 1, neuron] - self.rhos_oil[well, sep, 1, neuron] - quicksum(self.weights_oil[well][sep][0][dim][neuron]*self.inputs[well, sep, dim] for dim in range(self.multidims_oil[well][sep][0])) == self.biases_oil[well][sep][0][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(self.multidims_oil[well][sep][1]) )
        self.m.addConstrs(self.mus_oil[well, sep, layer, neuron] - self.rhos_oil[well, sep, layer, neuron] - quicksum((self.weights_oil[well][sep][layer-1][dim][neuron]*self.mus_oil[well, sep, layer-1, dim]) for dim in range(self.multidims_oil[well][sep][layer-1])) == self.biases_oil[well][sep][layer-1][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(2, self.layers_oil[well][sep]-1) for neuron in range(self.multidims_oil[well][sep][layer]) )
        
        # lambda indicator NN constraints        
        # =============================================================================
        #gas
        self.m.addConstrs( (self.lambdas_gas[scenario, well, sep, layer, neuron] == 1) >> (self.mus_gas[scenario, well, sep, layer, neuron] <= 0) for scenario in range(self.scenarios)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_gas[scenario][well][sep]-1) for neuron in range(self.multidims_gas[scenario][well][sep][layer]))
        self.m.addConstrs( (self.lambdas_gas[scenario, well, sep, layer, neuron] == 0) >> (self.rhos_gas[scenario, well, sep, layer, neuron] <= 0) for scenario in range(self.scenarios) for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1,self.layers_gas[scenario][well][sep]-1) for neuron in range(self.multidims_gas[scenario][well][sep][layer]))
        #oil
        self.m.addConstrs( (self.lambdas_oil[well, sep, layer, neuron] == 1) >> (self.mus_oil[well, sep, layer, neuron] <= 0)   for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_oil[well][sep]-1) for neuron in range(self.multidims_oil[well][sep][layer]))
        self.m.addConstrs( (self.lambdas_oil[well, sep, layer, neuron] == 0) >> (self.rhos_oil[well, sep, layer, neuron] <= 0)   for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_oil[well][sep]-1) for neuron in range(self.multidims_oil[well][sep][layer]))

        #use these to model last layer as linear instead of ReLU
        #gas
        self.m.addConstrs( (self.routes[well, sep] == 1) >> (self.outputs_gas[scenario, well, sep] ==  quicksum(self.weights_gas[scenario][well][sep][self.layers_gas[scenario][well][sep]-2][neuron][0] * self.mus_gas[scenario, well, sep, self.layers_gas[scenario][well][sep]-2, neuron] for neuron in range(self.multidims_gas[scenario][well][sep][self.layers_gas[scenario][well][sep]-2]) ) + self.biases_gas[scenario][well][sep][self.layers_gas[scenario][well][sep]-2][0]) for scenario in range(self.scenarios) for well in self.wellnames for sep in self.well_to_sep[well] )
        self.m.addConstrs( (self.routes[well, sep] == 0) >> (self.outputs_gas[scenario, well, sep] == 0) for scenario in range(self.scenarios) for well in self.wellnames for sep in self.well_to_sep[well])
        #oil
        self.m.addConstrs( (self.routes[well, sep] == 1) >> (self.outputs_oil[well, sep] == quicksum(self.weights_oil[well][sep][self.layers_oil[well][sep]-2][neuron][0] * self.mus_oil[well, sep, self.layers_oil[well][sep]-2, neuron] for neuron in range(self.multidims_oil[well][sep][self.layers_oil[well][sep]-2]) ) + self.biases_oil[well][sep][self.layers_oil[well][sep]-2][0]) for well in self.wellnames for sep in self.well_to_sep[well])
        self.m.addConstrs( (self.routes[well, sep] == 0) >> (self.outputs_oil[well, sep] == 0) for well in self.wellnames for sep in self.well_to_sep[well] )
        
        return 
    
    
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
        
        Recourse_Model.init(self, case, num_scen, lower, upper, phase, sep, save, store_init, init_name, max_changes, w_relative_change, stability_iter, distr, lock_wells, scen_const, recourse_iter)        
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
        # Variables mu and rho are indexed from 1 and up since the input layer is layer 0 in multidims. 
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
    
    
class SOS2(Recourse_Model):
    # =============================================================================
    # Function to build model with all wells in the problem.
    # =============================================================================
    def init(self, case=2,
                num_scen = 200, lower=-4, upper=4, phase="gas", sep="HP", save=True,store_init=False, init_name=None, 
                max_changes=15, w_relative_change=None, stability_iter=None, distr="truncnorm", lock_wells=None, scen_const=None, recourse_iter=False, verbose=1, points=10):
        
        Recourse_Model.init(self, case, num_scen, lower, upper, phase, sep, save, store_init, init_name, max_changes, w_relative_change, stability_iter, distr, lock_wells, scen_const, recourse_iter)        
        self.results_file = "results/robust/res_SOS2.csv"
        #load SOS2 breakpoints
        self.oil_vals = t.get_sos2_scenarios("oil", self.scenarios)
        self.gas_vals = t.get_sos2_scenarios("gas", self.scenarios)
        if(self.scenarios=="eev"):
            self.scenarios=1
        self.choke_vals = [i*100/(len(self.oil_vals["W1"])-1) for i in range(len(self.oil_vals["W1"]))]
        
        # =============================================================================
        # variable creation                    
        # =============================================================================
        #continuous breakpoint variables
        self.zetas = self.m.addVars([(brk, well, sep) for well in self.wellnames for sep in self.well_to_sep[well] for brk in range(len(self.choke_vals))], vtype = GRB.CONTINUOUS, name="zetas")

        # =============================================================================
        # SOS2
        # =============================================================================
        #oil
        self.m.addConstrs( (self.routes[well, sep] == 1) >> (self.outputs_oil[well, sep] == quicksum( self.zetas[brk, well, sep]*self.oil_vals[well][brk] for brk in range(len(self.choke_vals)))) for well in self.wellnames for sep in self.well_to_sep[well])
        self.m.addConstrs( (self.routes[well, sep] == 0) >> (self.outputs_oil[well, sep] == 0) for well in self.wellnames for sep in self.well_to_sep[well] )
        #gas
        self.m.addConstrs( (self.routes[well, sep] == 1) >> (self.outputs_gas[scenario, well, sep] == quicksum(self.zetas[brk, well, sep]*self.gas_vals[scenario][well][brk] for brk in range(len(self.choke_vals)))) for well in self.wellnames for sep in self.well_to_sep[well] for scenario in range(self.scenarios) )
        self.m.addConstrs( (self.routes[well, sep] == 0) >> (self.outputs_gas[scenario, well, sep] == 0) for scenario in range(self.scenarios) for well in self.wellnames for sep in self.well_to_sep[well] ) 
        # =============================================================================
        # input limit constraints
        # =============================================================================
        self.m.addConstrs( self.inputs[well, sep, 0] == quicksum(self.zetas[brk, well, sep]*self.choke_vals[brk] for brk in range(len(self.choke_vals))) for well in self.wellnames for sep in self.well_to_sep[well])
#        self.m.addConstrs( (self.routes[well, sep] == 0) >> (self.inputs[well, sep, 0] == 0) for well in self.wellnames for sep in self.well_to_sep[well])

        # =============================================================================
        # SOS2 constraints 
        # =============================================================================
        #no way to do in one-liner
        for well in self.wellnames:
            for sep in self.well_to_sep[well]:
                self.m.addSOS(2, [self.zetas[brk, well, sep] for brk in range(len(self.choke_vals))])
        #tighten sos constraints
        self.m.addConstrs( self.routes[well, sep] == quicksum( self.zetas[brk, well, sep] for brk in range(len(self.choke_vals))) for well in self.wellnames for sep in self.well_to_sep[well] for scenario in range(self.scenarios) )
        return self