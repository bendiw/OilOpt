# -*- coding: utf-8 -*f-

# =============================================================================
# Nomenclature:
#   Thesis  |   Code
#   x       |    mu
#   s       |    rho
#   z       |    lambda
# =============================================================================
from gurobipy import *
import numpy as np
import math
import pandas as pd
import tools as t
import os.path
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

    
    
    # =============================================================================
    # Function to run with all wells in the problem.
    # =============================================================================
    def init(self, case=2, load_M = False,
                num_scen = 1000, lower=-4, upper=4, phase="gas", sep="HP", save=True,store_init=False, init_name=None, 
                max_changes=15, w_relative_change=None, stability_iter=None, distr="truncnorm", lock_wells=None, scen_const=None, recourse_iter=False, verbose=1, points=10):
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
        
        
#        self.s_draw = t.get_scenario(case, num_scen, lower=lower, upper=upper,
#                                     phase=phase, sep=sep, iteration=stability_iter, distr=distr)
        self.scenarios = num_scen
        self.results_file = "results/robust/res_SOS2.csv"
        
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
        
        #load SOS2 breakpoints
        self.oil_vals = t.get_sos2_scenarios("oil", self.scenarios)
        self.gas_vals = t.get_sos2_scenarios("gas", self.scenarios)
        if(self.scenarios=="eev"):
            self.scenarios=1
        self.choke_vals = [i*100/(len(self.oil_vals["W1"])-1) for i in range(len(self.oil_vals["W1"]))]
#        print(self.oil_vals)
        #workaround to multidims
        self.multidims= {}

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
        input_upper = {(well, sep, dim) : w_max_lims[dim][well][sep]  for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1)}
        input_lower = {(well, sep, dim) : w_min_lims[dim][well][sep]  for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1)}
        
        
        # =============================================================================
        # variable creation                    
        # =============================================================================
        #keep input dummy variables to track changes more easily
        self.inputs = self.m.addVars(input_upper.keys(), ub = input_upper, lb=input_lower, name="input", vtype=GRB.SEMICONT) #SEMICONT
        
        #continuous breakpoint variables
        self.zetas = self.m.addVars([(brk, well, sep) for well in self.wellnames for sep in self.well_to_sep[well] for brk in range(len(self.choke_vals))], vtype = GRB.CONTINUOUS, name="zetas")

        #routing and change tracking        
        self.routes = self.m.addVars([(well, sep) for well in self.wellnames for sep in self.well_to_sep[well]], vtype = GRB.BINARY, name="routing")
        self.changes = self.m.addVars([(well, sep, dim) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1)], vtype=GRB.BINARY, name="changes")

        #new variables to control routing decision and input/output
        self.outputs_gas = self.m.addVars([(scenario, well, sep) for well in self.wellnames for sep in self.well_to_sep[well] for scenario in range(self.scenarios)], vtype = GRB.CONTINUOUS, name="outputs_gas")
        self.outputs_oil = self.m.addVars([(well, sep) for well in self.wellnames for sep in self.well_to_sep[well]], vtype = GRB.CONTINUOUS, name="outputs_oil")



        # =============================================================================
        # Oil output SOS2
        # =============================================================================
        self.m.addConstrs( (self.routes[well, sep] == 1) >> (self.outputs_oil[well, sep] == quicksum( self.zetas[brk, well, sep]*self.oil_vals[well][brk] for brk in range(len(self.choke_vals)))) for well in self.wellnames for sep in self.well_to_sep[well])
        self.m.addConstrs( (self.routes[well, sep] == 0) >> (self.outputs_oil[well, sep] == 0) for well in self.wellnames for sep in self.well_to_sep[well] )

        # =============================================================================
        # Gas output SOS2        
        # =============================================================================
        self.m.addConstrs( (self.routes[well, sep] == 1) >> (self.outputs_gas[scenario, well, sep] == quicksum(self.zetas[brk, well, sep]*self.gas_vals[scenario][well][brk] for brk in range(len(self.choke_vals)))) for well in self.wellnames for sep in self.well_to_sep[well] for scenario in range(self.scenarios) )
        self.m.addConstrs( (self.routes[well, sep] == 0) >> (self.outputs_gas[scenario, well, sep] == 0) for scenario in range(self.scenarios) for well in self.wellnames for sep in self.well_to_sep[well] ) 

        # =============================================================================
        # input limit constraints
        # =============================================================================
        self.m.addConstrs( self.inputs[well, sep, 0] == quicksum(self.zetas[brk, well, sep]*self.choke_vals[brk] for brk in range(len(self.choke_vals))) for well in self.wellnames for sep in self.well_to_sep[well])
        self.m.addConstrs( (self.routes[well, sep] == 0) >> (self.inputs[well, sep, 0] == 0) for well in self.wellnames for sep in self.well_to_sep[well])

        # =============================================================================
        # SOS2 constraints 
        # =============================================================================
        #no way to do in one-liner
        for well in self.wellnames:
            for sep in self.well_to_sep[well]:
                self.m.addSOS(2, [self.zetas[brk, well, sep] for brk in range(len(self.choke_vals))])
        #tighten sos constraints
        self.m.addConstrs( self.routes[well, sep] == quicksum( self.zetas[brk, well, sep] for brk in range(len(self.choke_vals))) for well in self.wellnames for sep in self.well_to_sep[well] for scenario in range(self.scenarios) )

        
        # =============================================================================
        # separator gas constraints
        # =============================================================================
        if(case==1):
            lp_constr = self.m.addConstr(quicksum(outputs[well, "gas", "LP"] for p in sep_p_route["LP"] for well in p_dict[p]) - quicksum(input_dummies[c_well, "LP", 0] for c_well in p_dict["C"]) <= sep_cap["LP"])
    
    #        lp_constr = self.m.addConstr(quicksum(outputs[well, "gas", "LP"] for p in sep_p_route["LP"] for well in p_dict[p]) - quicksum(inputs[c_well, "LP", 0] for c_well in p_dict["C"]) <= sep_cap["LP"])
            hp_constr = self.m.addConstr(quicksum(outputs[well, "gas", "HP"] for p in sep_p_route["HP"] for well in p_dict[p]) <= sep_cap["HP"])
        else:
            #single gas constraint per well in case2
            gas_constr = self.m.addConstrs(outputs_gas[scenario, well, "HP"] <= self.well_cap for well in self.wellnames for scenario in range(self.scenarios))
            
    
        # =============================================================================
        # total gas export, robust constraints
        # =============================================================================
        if(case==1):
            exp_constr = self.m.addConstr(quicksum(outputs[well, "gas", sep] for well in self.wellnames for sep in self.well_to_sep[well]) - quicksum(input_dummies[c_well, "LP", 0] for c_well in p_dict["C"]) <= tot_exp_cap)
#        exp_constr = self.m.addConstr(quicksum(outputs[well, "gas", sep] for well in self.wellnames for sep in self.well_to_sep[well]) - quicksum(inputs[c_well, "LP", 0] for c_well in p_dict["C"]) <= tot_exp_cap)
        else:
            exp_constr = self.m.addConstrs(quicksum(self.outputs_gas[scenario, well, sep] for well in self.wellnames for sep in self.well_to_sep[well]) <= self.tot_exp_cap for scenario in range(self.scenarios))
        
        # =============================================================================
        # routing
        # =============================================================================
        self.m.addConstrs(quicksum(routes[well, sep] for sep in self.well_to_sep[well]) <= 1 for well in self.wellnames)

    
        
        # =============================================================================
#         change tracking and total changes
#         =============================================================================
        self.m.addConstrs(self.w_initial_vars[well][dim] - self.inputs[well, sep, dim] <= self.changes[well, sep, dim]*self.w_initial_vars[well][dim]*w_relative_change[well][dim] + 
                          (self.w_initial_vars[well][dim]*(1-quicksum(self.routes[well, separ] for separ in self.well_to_sep[well]))*(1-w_relative_change[well][dim])) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1))
        
        self.m.addConstrs(self.inputs[well, sep, dim] - self.w_initial_vars[well][dim] <= self.changes[well, sep, dim]*self.w_initial_vars[well][dim]*w_relative_change[well][dim]+
                          (1-self.w_initial_prod[well])*w_max_lims[dim][well][sep]*self.changes[well, sep, dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1))
        
        self.m.addConstr(quicksum(self.changes[well, sep, dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1)) <= max_changes)


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
        self.m.setObjective( quicksum(outputs_oil[well, sep] for well in self.wellnames for sep in self.well_to_sep[well]), GRB.MAXIMIZE)

        self.m.optimize()
        
    def get_solution(self):
        df = pd.DataFrame(columns=t.robust_res_columns_SOS2) 
        chokes = [sum(self.zetas[brk, well, "HP"].x*self.choke_vals[brk] for brk in range(len(self.choke_vals)))  if outputs_gas[0, well, "HP"].x>0 else 0 for well in self.wellnames]
        rowlist=[]
        if(self.case==2 and self.save):
            gas_mean = np.zeros(len(self.wellnames))
#            gas_var=[]
            w = 0
            for well in self.wellnames:
                for scenario in range(self.scenarios):
                    gas_mean[w] += self.outputs_gas[scenario, well, "HP"].x

                w += 1
            gas_mean = (gas_mean/float(self.scenarios)).tolist()
            oil_mean = [self.outputs_oil[well, "HP"].x for well in self.wellnames]
#            oil_var = [outputs_]
            tot_oil = sum(oil_mean)
            tot_gas = sum(gas_mean)
            change = [abs(self.changes[w, "HP", 0].x) for w in self.wellnames]
#            print(df.columns)
            rowlist = [self.scenarios, self.tot_exp_cap, self.well_cap, tot_oil, tot_gas]+chokes+gas_mean+oil_mean+change
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
            gas_var = []
            for well in self.wellnames:
                pass
            tot_oil = sum(oil_mean)
            tot_gas = sum(gas_mean)+sum([gas_var[w]*self.s_draw.loc[s][self.wellnames[w]] for w in range(len(self.wellnames)) for s in range(self.scenarios)])/self.scenarios
            change = [abs(self.changes[w, "HP", 0].x) for w in self.wellnames]
            rowlist = [self.tot_exp_cap, self.well_cap, tot_oil, tot_gas]+chokes+gas_mean+oil_mean+gas_var+change
        return rowlist

        
        
    def nn(self, well, sep, load_M = False):
        self.wellnames = [well]
        self.well_to_sep[well]= [sep]
        self.platforms= [well[0]]
        self.p_dict[well[0]] = [well]
        self.p_sep_names[self.platforms[0]] = [sep]
        self.phasenames = ["oil", "gas"]
        self.run(load_M=load_M)