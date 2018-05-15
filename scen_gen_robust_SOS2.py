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
    # get breakpoints for scenario curves    
    # =============================================================================
    def getBreakpoints(self, case, points, scenarios, phase):
        if(phase=="oil"):
            choke_vals = {w:[] for w in self.wellnames}
            out_vals = {w:[] for w in self.wellnames}
            for well in self.wellnames:
                #TODO: load values
                pass
        else:
            choke_vals = {s : {w:[] for w in self.wellnames} for s in range(scenarios)}
            out_vals = {s : {w:[] for w in self.wellnames} for s in range(scenarios)}
            for s in range(scenarios):
                for well in self.wellnames:
                    #TODO: load values here
                    pass
        return choke_vals, out_vals
        
    # =============================================================================
    # Wrapper to evaluate EEV versus scenario based optimization    
    # =============================================================================
    def evaluate_recourse(self, iterations=201, num_scen=100, init_name="zero", max_changes=3, verbose=0):
        scen_draws = t.get_scenario(case=2, num_scen=iterations, distr="truncnorm")
        results = np.zeros((iterations,2))
        
        #get basic solution
        base_sol = self.recourse_opt(init_name=init_name, max_changes=max_changes, num_scen=num_scen, pregen_scen=scen_draws.loc[0].to_dict(), verbose=verbose, base_solution=None)
        #now iterate with max_changes-1 since basic solution used max_changes
        for i in range(iterations):
            print("iteration:", i)
            results[i] = self.recourse_opt(init_name=init_name, max_changes=max_changes-1, num_scen=num_scen, pregen_scen=scen_draws.loc[i].to_dict(), verbose=verbose, base_solution=base_sol)
        res_df = pd.DataFrame(results, columns=["infeasible_count", "total_oil"])
        print(res_df)
        res_df.to_csv("results/robust_recourse_iterative/results_"+init_name+str(num_scen)+"_scen_"+str(iterations)+"_iter.csv", sep=';', header=True)
    
    # =============================================================================
    # Iteratively optimize with recourse options. A solution is implemented by
    # following the first step of an optimization result possibly involving
    # several wells. The scenario is drawn and locked for the implemented well.
    # We then re-optimize.
    #
    # The base_solution flag is used to obtain a first solution since it will
    # be equal for all pregen_scen
    # =============================================================================
    def recourse_opt(self, init_name, max_changes, num_scen, pregen_scen=None, verbose=1, base_solution=None):
#        results = pd.DataFrame(columns=t.robust_res_columns) 
        scen_draws = t.get_scenario(case=2, num_scen=10000, distr="truncnorm")
        scen_const = {}
        lock_wells = []
        implemented = t.get_robust_solution(init_name=init_name)
        implemented.columns = t.robust_res_columns_recourse
#        print("oil contrib:", current_contrib_oil)
#        print("gas contrib:", current_contrib_gas)

        change_well = None
        chokenames = [w+"_choke" for w in t.wellnames_2]
        infeasible_count= 0
        infeasible = False
        #get starting point
        results= t.get_robust_solution(init_name=init_name)
        results.columns = t.robust_res_columns_recourse
        current_contrib_gas = {w:results[w+"_gas_mean"].values[0]+pregen_scen[w]*results[w+"_gas_var"].values[0] for w in t.wellnames_2}
        current_contrib_oil = {w:results[w+"_oil_mean"].values[0] for w in t.wellnames_2}
        init_choke = results.loc[0][chokenames].to_dict()
        if base_solution is not None:
            #solution was given as argument
            #we now check if it is feasible given the pregen_scen
            z = np.zeros((1,implemented.shape[1]))
            implemented.loc[implemented.shape[0]] = z[0]
            
            #greedily implement 1 change from solution
            found = False
            #start by decrementing in reverse order
            for w in t.well_order[::-1]:
                if(base_solution[w+"_changed"]>0 and round(base_solution[w+"_choke"], 4)-round(init_choke[w+"_choke"],4)<0):
                    if(w not in lock_wells):
                        lock_wells.append(w)
                        if not pregen_scen:
                            s = np.random.randint(0, 10000)
                            scen_const[w] = scen_draws.loc[s][w]
                        else:
                            scen_const[w] = pregen_scen[w]
                    change_well = w
                    found = True
                    break
            if not found:
                for w in t.well_order:
                    if(base_solution[w+"_changed"]>0 and base_solution[w+"_choke"]-init_choke[w+"_choke"]>0.01): # and results.loc[iter_num]
                        if(w not in lock_wells):
                            lock_wells.append(w)
                            if not pregen_scen:
                                s = np.random.randint(0, 10000)
                                scen_const[w] = scen_draws.loc[s][w]
                            else:
                                scen_const[w] = pregen_scen[w]
                        change_well = w
                        break
            new_indiv_gas = base_solution[change_well+"_gas_mean"]+scen_const[change_well]*base_solution[change_well+"_gas_var"]
            new_indiv_oil = base_solution[change_well+"_oil_mean"]
            new_tot_gas = implemented.loc[0]["tot_gas"]+new_indiv_gas - current_contrib_gas[change_well]
            infeasible = (round(new_indiv_gas, 2) > results.loc[0]["indiv_cap"] or new_tot_gas>results.loc[0]["tot_cap"])

            if(infeasible):
                #revert solution
                implemented.loc[implemented.shape[0]] = implemented.loc[0]
                results.loc[results.shape[0]-1] = results.loc[0]
                infeasible_count+=1
            else:
                current_contrib_gas[change_well] = new_indiv_gas
                results.loc[results.shape[0]] = base_solution
                implemented.loc[implemented.shape[0]-1]["tot_gas"] = new_tot_gas
                current_contrib_oil[change_well] = new_indiv_oil
                implemented.loc[implemented.shape[0]-1]["tot_oil"] = sum(current_contrib_oil.values())
                implemented.loc[implemented.shape[0]-1][change_well+"_choke"] = base_solution[change_well+"_choke"]
                implemented.loc[implemented.shape[0]-1][change_well+"_gas_mean"] = new_indiv_gas
                init_choke[change_well+"_choke"] = base_solution[change_well+"_choke"]

                
#            print("impl pre-iteration start:\n", implemented)
        #iterate down to 1 change allowed
        for i in range(max_changes, 0, -1):
            iter_num = max_changes+2-i
            if(change_well and not infeasible):
                #keep track of choke settings
                init_choke[change_well+"_choke"] = results.loc[iter_num-1][change_well+"_choke"]

            #optimize
            res = self.init(num_scen=num_scen, save=False, init_name=init_choke, max_changes=i, recourse_iter=True, 
                            scen_const=scen_const, verbose=0)
            results.loc[iter_num] = res
            if base_solution is None:
                #return base solution
                return results.loc[iter_num]
            z = np.zeros((1,implemented.shape[1]))
            implemented.loc[iter_num] = z[0]
            
            #greedily implement 1 change from solution
            found = False
            #start by decrementing in reverse order
            for w in t.well_order[::-1]:
                if(results.loc[iter_num][w+"_changed"]>0 and round(results.loc[iter_num][w+"_choke"], 4)-round(init_choke[w+"_choke"],4)<0):
                    if(w not in lock_wells):
                        lock_wells.append(w)
                        if not pregen_scen:
                            s = np.random.randint(0, 10000)
                            scen_const[w] = scen_draws.loc[s][w]
                        else:
                            scen_const[w] = pregen_scen[w]
                    change_well = w
                    found = True
                    break
            if not found:
                for w in t.well_order:
                    if(results.loc[iter_num][w+"_changed"]>0 and results.loc[iter_num][w+"_choke"]-init_choke[w+"_choke"]>0.01): # and results.loc[iter_num]
                        if(w not in lock_wells):
                            lock_wells.append(w)
                            if not pregen_scen:
                                s = np.random.randint(0, 10000)
                                scen_const[w] = scen_draws.loc[s][w]
                            else:
                                scen_const[w] = pregen_scen[w]
                        change_well = w
                        break
            new_indiv_gas = results.loc[iter_num][change_well+"_gas_mean"]+scen_const[change_well]*results.loc[iter_num][change_well+"_gas_var"]
            new_indiv_oil = results.loc[iter_num][change_well+"_oil_mean"]
            new_tot_gas = implemented.loc[iter_num-1]["tot_gas"]+new_indiv_gas - current_contrib_gas[change_well]
            infeasible = (round(new_indiv_gas, 2) > results.loc[0]["indiv_cap"] or new_tot_gas>results.loc[0]["tot_cap"])

            if(infeasible):
                #revert solution
                implemented.loc[iter_num] = implemented.loc[iter_num-1]
                infeasible_count+=1
            else:
                current_contrib_gas[change_well] = new_indiv_gas

                implemented.loc[iter_num]["tot_gas"] = new_tot_gas
                current_contrib_oil[change_well] = new_indiv_oil
                implemented.loc[iter_num]["tot_oil"] = sum(current_contrib_oil.values())
                implemented.loc[iter_num][change_well+"_choke"] = results.loc[iter_num][change_well+"_choke"]
                implemented.loc[iter_num][change_well+"_gas_mean"] = new_indiv_gas
            if(verbose>0):
                print("\n\niteration", iter_num)
#                for w in chokenames:
#                    print(w, "change:", round(results.loc[iter_num][w], 4)-round(init_choke[w],4))
                print("initial choke settings:", init_choke)
                print("scenario draws:", scen_const)
                print("well to change:", change_well, "-->", results.loc[iter_num][change_well+"_choke"])
                print("indiv gas:", new_indiv_gas, "\ntot_gas", new_tot_gas, "\ninfeasible?", infeasible)
        if(verbose>0):
            print("\n\nImplemented:\n", implemented[implemented.columns[2:11]])
            if(verbose>1):
                for c in results.columns:
                    print(results[c])
        return [infeasible_count, implemented.loc[implemented.shape[0]-1]["tot_oil"]]
    
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
        
        self.s_draw = t.get_scenario(case, num_scen, lower=lower, upper=upper,
                                     phase=phase, sep=sep, iteration=stability_iter, distr=distr)
        self.scenarios = len(self.s_draw)
        self.results_file = "results/robust/res.csv"
        
        #alternative results file for storing init solutions
        self.results_file_init = "results/initial/res_initial.csv"

        self.phasenames = t.phasenames
            
        #Case relevant numerics
        if case==2:
            if(not w_relative_change):
                w_relative_change = {well : [0.4] for well in self.wellnames}
            
            if(lock_wells):
                #wells which are not allowed to change
                assert(isinstance(lock_wells, (list)))
                for w in lock_wells:
                    w_relative_change[w] = [0, 0]
                    
            if(scen_const):
                #provide "locked" scenario constant for certain wells
                assert(isinstance(scen_const, (dict)))
                for well, val in scen_const.items():
                    self.s_draw[well] = val
            
            if not init_name:
                w_initial_prod = {well : 0 for well in self.wellnames}
                w_initial_vars = {well : [0] for well in self.wellnames}
            elif not isinstance(init_name, (dict)):
                #load init from file
                w_initial_df = t.get_robust_solution(init_name=init_name)
                w_initial_vars = {w:[w_initial_df[w+"_choke"].values[0]] for w in self.wellnames}
                print(w_initial_vars)
                
                w_initial_prod = {well : 1. if w_initial_vars[well][0]>0 else 0. for well in self.wellnames}
                print(w_initial_prod)
            #constraints for case 2
            else:
                #we were given a dict of initial values
#                w_initial_vars=init_name
                w_initial_vars={}
                for w in self.wellnames:
                    w_initial_vars[w] = [init_name[w+"_choke"]]
#                    del w_initial_vars[w+"_choke"]
                
#                print("optimization initial chokes:", w_initial_vars)
                w_initial_prod = {well : 1 if w_initial_vars[well][0]>0 else 0 for well in self.wellnames}
            tot_exp_cap = 250000
            well_cap = 54166
        else:
            raise ValueError("Case 1 not implemented yet.")
            
        # =============================================================================
        # initialize an optimization model
        # =============================================================================
        self.m = Model("Model")
        
        #load SOS2 breakpoints
        self.choke_vals, self.oil_vals = self.getBreakpoints(case=case, points=points, scenarios=self.scenarios, phase="oil")
        _, self.gas_vals = self.getBreakpoints(case=case, points=points, scenarios=self.scenarios, phase="gas")
        
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
        inputs = self.m.addVars(input_upper.keys(), ub = input_upper, lb=input_lower, name="input", vtype=GRB.SEMICONT) #SEMICONT
        
        #continuous breakpoint variables
        zetas = self.m.addVars([(brk, well, sep) for well in self.wellnames for sep in self.well_to_sep[well] for brk in range(len(self.choke_vals[well]))], vtype = GRB.CONTINUOUS, name="zetas")

        #routing and change tracking        
        routes = self.m.addVars([(well, sep) for well in self.wellnames for sep in self.well_to_sep[well]], vtype = GRB.BINARY, name="routing")
        changes = self.m.addVars([(well, sep, dim) for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep][0])], vtype=GRB.BINARY, name="changes")

        #new variables to control routing decision and input/output
        outputs_gas = self.m.addVars([(scenario, well, sep) for well in self.wellnames for sep in self.well_to_sep[well] for scenario in range(self.scenarios)], vtype = GRB.CONTINUOUS, name="outputs_gas")
        outputs_oil = self.m.addVars([(well, sep) for well in self.wellnames for sep in self.well_to_sep[well]], vtype = GRB.CONTINUOUS, name="outputs_oil")




        # =============================================================================
        # Oil output SOS2
        # =============================================================================
        self.m.addConstrs( (routes[well, sep] == 1) >> (outputs_oil[well, sep] == quicksum( zetas[brk, well, sep]*self.oil_vals[well][brk] for brk in range(len(self.choke_vals[well])))) for well in self.wellnames for sep in self.well_to_sep[well])
        self.m.addConstrs( (routes[well, sep] == 0) >> (outputs_oil[well, sep] == 0) for well in self.wellnames for sep in self.well_to_sep[well] )

        # =============================================================================
        # Gas output SOS2        
        # =============================================================================
        self.m.addConstrs( (routes[well, sep] == 1) >> (outputs_gas[scenario, well, sep] == quicksum( zetas[brk, well, sep]*self.gas_vals[scenario][well][brk] for brk in range(len(self.choke_vals)))) for well in self.wellnames for sep in self.well_to_sep[well] for scenario in range(self.scenarios) )
        self.m.addConstrs( (routes[well, sep] == 0) >> (outputs_gas[scenario, well, sep] == 0) for scenario in range(self.scenarios) for well in self.wellnames for sep in self.well_to_sep[well] ) 

        # =============================================================================
        # SOS2 constraints 
        # =============================================================================
        #no way to do in one-liner
        for well in self.wellnames:
            for sep in self.well_to_sep[well]:
                self.m.addSOS(2, [zetas[brk, well, sep] for brk in range(len(self.choke_vals[well]))])
        #tighten sos constraints
        self.m.addConstrs( routes[well, sep] == quicksum( zetas[brk, well, sep] for brk in range(len(self.choke_vals[well]))) for well in self.wellnames for sep in self.well_to_sep[well] for scenario in range(self.scenarios) )

        
        # =============================================================================
        # separator gas constraints
        # =============================================================================
        if(case==1):
            lp_constr = self.m.addConstr(quicksum(outputs[well, "gas", "LP"] for p in sep_p_route["LP"] for well in p_dict[p]) - quicksum(input_dummies[c_well, "LP", 0] for c_well in p_dict["C"]) <= sep_cap["LP"])
    
    #        lp_constr = self.m.addConstr(quicksum(outputs[well, "gas", "LP"] for p in sep_p_route["LP"] for well in p_dict[p]) - quicksum(inputs[c_well, "LP", 0] for c_well in p_dict["C"]) <= sep_cap["LP"])
            hp_constr = self.m.addConstr(quicksum(outputs[well, "gas", "HP"] for p in sep_p_route["HP"] for well in p_dict[p]) <= sep_cap["HP"])
        else:
            #single gas constraint per well in case2
            gas_constr = self.m.addConstrs(outputs_gas[scenario, well, "HP"] <= well_cap for well in self.wellnames for scenario in range(self.scenarios))
            
    
        # =============================================================================
        # total gas export, robust constraints
        # =============================================================================
        if(case==1):
            exp_constr = self.m.addConstr(quicksum(outputs[well, "gas", sep] for well in self.wellnames for sep in self.well_to_sep[well]) - quicksum(input_dummies[c_well, "LP", 0] for c_well in p_dict["C"]) <= tot_exp_cap)
#        exp_constr = self.m.addConstr(quicksum(outputs[well, "gas", sep] for well in self.wellnames for sep in self.well_to_sep[well]) - quicksum(inputs[c_well, "LP", 0] for c_well in p_dict["C"]) <= tot_exp_cap)
        else:
            exp_constr = self.m.addConstrs(quicksum(outputs_gas[scenario, well, sep] for well in self.wellnames for sep in self.well_to_sep[well]) <= tot_exp_cap for scenario in range(self.scenarios))
        
        # =============================================================================
        # routing
        # =============================================================================
        self.m.addConstrs(quicksum(routes[well, sep] for sep in self.well_to_sep[well]) <= 1 for well in self.wellnames)

    
        
        # =============================================================================
#         change tracking and total changes
#         =============================================================================
        self.m.addConstrs(w_initial_vars[well][dim] - inputs[well, sep, dim] <= changes[well, sep, dim]*w_initial_vars[well][dim]*w_relative_change[well][dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep][0]))
        
        self.m.addConstrs(inputs[well, sep, dim] - w_initial_vars[well][dim] <= changes[well, sep, dim]*w_initial_vars[well][dim]*w_relative_change[well][dim]+
                          (1-w_initial_prod[well])*w_max_lims[dim][well][sep]*changes[well, sep, dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well]["oil"][sep][0]))
        
        self.m.addConstr(quicksum(changes[well, sep, dim] for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(1)) <= max_changes)

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
        
        # =============================================================================
        # temporary constraints to generate initial scenarios        
        # =============================================================================
#        self.m.addConstr(quicksum(outputs_oil[well, sep] for well in self.wellnames for sep in self.well_to_sep[well]) <= 90.0)
#        self.m.addConstr(outputs_oil["W3", "HP"] ==0)
#        self.m.addConstr(outputs_oil["W1", "HP"] ==0)

        #maximization of mean oil. no need to take mean over scenarios since only gas is scenario dependent
        self.m.setObjective( quicksum(outputs_oil[well, sep] for well in self.wellnames for sep in self.well_to_sep[well]), GRB.MAXIMIZE)
        
        

        self.m.optimize()
        
        df = pd.DataFrame(columns=t.robust_res_columns) 
        chokes = [inputs[well, "HP", 0].x if outputs_gas[0, well, "HP"].x>0 else 0 for well in self.wellnames]
        rowlist=[]
        if(case==2 and save):
            gas_mean = np.zeros(len(self.wellnames))
            gas_var=[]
            w = 0
            for well in self.wellnames:
                for scenario in range(self.scenarios):
                    gas_mean[w] += outputs_gas[scenario, well, "HP"].x

                w += 1
            gas_mean = (gas_mean/float(self.scenarios)).tolist()
            oil_mean = [outputs_oil[well, "HP"].x for well in self.wellnames]
#            oil_var = [outputs_]
            tot_oil = sum(oil_mean)
            tot_gas = sum(gas_mean)
            change = [abs(changes[w, "HP", 0].x) for w in self.wellnames]
#            print(df.columns)
            rowlist = [self.scenarios, tot_exp_cap, well_cap, tot_oil, tot_gas]+chokes+gas_mean+oil_mean+gas_var+change
#            print(len(rowlist))
            if(store_init):
                df.rename(columns={"scenarios": "name"}, inplace=True)
                rowlist[0] = init_name
                df.loc[df.shape[0]] = rowlist

                head = not os.path.isfile(self.results_file_init)
                with open(self.results_file_init, 'a') as f:
                    df.to_csv(f, sep=';', index=False, header=head)
            else:
                df.loc[df.shape[0]] = rowlist
                head = not os.path.isfile(self.results_file)
                with open(self.results_file, 'a') as f:
                    df.to_csv(f, sep=';', index=False, header=head)
        elif recourse_iter:
            oil_mean = [outputs_oil[well, "HP"].x for well in self.wellnames]
            gas_mean = []
            gas_var = []
            for well in self.wellnames:
                pass
            tot_oil = sum(oil_mean)
            tot_gas = sum(gas_mean)+sum([gas_var[w]*self.s_draw.loc[s][self.wellnames[w]] for w in range(len(self.wellnames)) for s in range(self.scenarios)])/self.scenarios
            change = [abs(changes[w, "HP", 0].x) for w in self.wellnames]
            rowlist = [tot_exp_cap, well_cap, tot_oil, tot_gas]+chokes+gas_mean+oil_mean+gas_var+change
        return rowlist

        
        
    def nn(self, well, sep, load_M = False):
        self.wellnames = [well]
        self.well_to_sep[well]= [sep]
        self.platforms= [well[0]]
        self.p_dict[well[0]] = [well]
        self.p_sep_names[self.platforms[0]] = [sep]
        self.phasenames = ["oil", "gas"]
        self.run(load_M=load_M)