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
from sys import stdout

from recourse_models import NN, SOS2, Factor
   

# =============================================================================
# The recourse algorithm function.
# Verbose levels:
# 0 - see only iteration number
# 1 - see basic information about starting scenario
# 2 - see iteration specific information
# 3 - see gurobi solver prints
#Use verbose=3 to see gurobi output
# =============================================================================
def recourse(num_iter=200, num_scen=10, max_changes=3, init_name=None, model_type="sos2", verbose=0, save=False):
    
    filestring = "results/robust_recourse_iterative/"+str(num_iter)+"iter_"+str(num_scen)+"scen_"+init_name+"_"+model_type+".csv"
    #init model
    model = get_model(model_type).init(max_changes=max_changes, num_scen=num_scen, init_name=init_name)
    
    
    #this is our starting point in terms of chokes
    init_chokes = t.get_init_chokes(init_name)
    if(verbose>0):
        print("Recourse from initial scenario", "'"+init_name+"'")
        print("Initial choke settings:")
        for w in t.wellnames_2:
            print(w, "\t",init_chokes[w])
    
    #store results in a dataframe
    results = pd.DataFrame(columns=t.recourse_algo_columns)
    
    #get first solution since it will be same for all iterations
    model.solve(verbose=max(verbose-2, 0))
    first_sol = model.get_chokes()
    
    
    for i in range(num_iter):
        stdout.write("\r iteration %d" % i)
        stdout.flush()

        #get "true" sos2 or NN models
        true_well_curves = get_true_models(init_name, i)
        #iterate with one less change since we have already found first solution
        results.loc[i] = iteration(model, init_chokes, first_sol, max_changes-1, true_well_curves, verbose=verbose)
        
        #revert to initial model
        model.set_chokes(first_sol)
        model.reset_m()
        model.m.update()
#        print(model.oil_out_constr)
    if(save):
        results.to_csv(filestring, sep=';')
    print("\n")
    return results

# =============================================================================
# Perform one iteration of algorithm
# =============================================================================
def iteration(model, init_chokes, first_sol, changes, true_well_curves, verbose=0):
    #store intermediary results as optimization finds them
#    opt_results = model.get_solution()
    #store implemented chokes, first we use only the starting point
    impl_chokes = init_chokes
    infeasible_count = 0
    tot_oil = sum([true_well_curves[w].predict(init_chokes[w], "oil") for w in t.wellnames_2])
    tot_gas = sum([true_well_curves[w].predict(init_chokes[w], "gas") for w in t.wellnames_2])
#    print("initchoke", init_chokes, "oil", tot_oil, "\ngas", tot_gas)
    new_chokes = first_sol
    
    #since we start by checking for #changes+1, we need to iterate an 'extra' time including 0
    for c in range(changes, -1, -1):
        inf_single, tot_oil, tot_gas, impl_chokes, change_well, change_gas = check_and_impl_change(true_well_curves, tot_oil, tot_gas, impl_chokes, new_chokes, model.well_cap, model.tot_exp_cap)
        infeasible_count+=inf_single
        if change_well is None:
            if verbose>1:
                print("No more suggested changes.")
            return [infeasible_count, tot_oil, tot_gas]+ list(impl_chokes.values())
        if(verbose>1):
            print("\n==================================================")
            print("changes:", c+1, "\nchange well:", change_well, "   [", (round(init_chokes[change_well],2) if c==changes else round(old_chokes[change_well],2)), "--->", round(new_chokes[change_well],2), "]")
            print("Infeasible! " if inf_single==1 else "Feasible.", change_well+" gas: "+str(round(change_gas,2)))
            print("\nimpl chokes:\t\t\tsuggested chokes:")
            for w in t.wellnames_2:
                print(w+"\t", round(impl_chokes[w],2), "\t\t\t", round(new_chokes[w],2))
            print("\ntot oil:\t", tot_oil, "\ntot gas:\t", tot_gas)
            print("==================================================\n")

#        print("res\n", opt_results)
        old_chokes = {key: value for key, value in impl_chokes.items()}
        model.set_chokes(impl_chokes)
        model.set_changes(c)
        if(c<1):
            break
#        model.set_changes(changes)

        model.set_true_curve(change_well, true_well_curves[change_well])
        model.solve(verbose=max(verbose-2, 0))
#        new_sol = model.get_solution()
#        opt_results.append(new_sol.to_dict(), ignore_index=True)
#        opt_results.loc[changes+1-c] = new_sol.loc[0].values.tolist()
        new_chokes = model.get_chokes()
    rowlist = [infeasible_count, tot_oil, tot_gas]+ list(impl_chokes.values())
    return rowlist
    
    
# =============================================================================
# Evaluate a solution given the true well curves and check infeasibility,
# implement a change if feasible
# =============================================================================
def check_and_impl_change(true_well_curves, tot_oil, tot_gas, old_chokes, new_chokes, indiv_cap, tot_cap):
#    tot_gas = 0
    found=False
    #find which well to change. prioritize negative changes
    for w in reversed(t.well_order):
        if(new_chokes[w] - old_chokes[w] < -0.01):
            change_well = w
            found=True
            break
    if not found:
        for w in t.well_order:
            if(abs(new_chokes[w]-old_chokes[w]) >= 0.01):
                change_well = w
                found=True
                break
    if not found:
        return 0, tot_oil, tot_gas, old_chokes, None, None
    #implement change and check for infeasibility
    temp_chokes = {key: value for key, value in old_chokes.items()}
    temp_chokes[change_well] = new_chokes[change_well]
#    print("\nchangewell:", change_well)
    old_gas = true_well_curves[change_well].predict(old_chokes[change_well], "gas")
    change_gas = true_well_curves[change_well].predict(temp_chokes[change_well], "gas")
    old_oil = true_well_curves[change_well].predict(old_chokes[change_well], "oil")
    change_oil = true_well_curves[change_well].predict(temp_chokes[change_well], "oil")
    if(change_gas-0.01 > indiv_cap[change_well] or tot_gas+(change_gas-old_gas)-0.01 > tot_cap):
        #infeasible, revert
#        print("tot", tot_gas+(change_gas-old_gas), "\tindiv:", change_gas)
        return 1, tot_oil, tot_gas, old_chokes, change_well, change_gas
    else:
        return 0, tot_oil+(change_oil-old_oil), tot_gas+(change_gas-old_gas), temp_chokes, change_well, change_gas
    
    
def test_init_scen(init_name, num_scen, phase="gas"):
    init_chokes = t.get_init_chokes(init_name)
    c = ["tot_"+phase, "infeasible"] + [w for w in t.wellnames_2]
    gas = {col:[] for col in c}
    for s in range(num_scen):
        stdout.write("\r scenario %d" % s)
        stdout.flush()
        true_well_curves = get_true_models(init_name, s)
        tot_gas = 0
        inf = 0
        for w in t.wellnames_2:
            w_gas =  true_well_curves[w].predict(init_chokes[w], phase)
            gas[w].append(w_gas)
            tot_gas+=w_gas
            if(w_gas-0.01 > indiv_cap):
                inf=1
        if(tot_gas-0.01 > tot_cap):
            inf=1
        gas["infeasible"].append(inf)
        gas["tot_"+phase].append(tot_gas)
    df = pd.DataFrame(gas, columns=c)
#    df.loc[0] = gas
    return df
    
# =============================================================================
# return correct model type
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
        
# =============================================================================
# get "true" well curves
# =============================================================================
def get_true_models(init_name, iteration, m_type = "sos2"):
    if(m_type=="sos2"):
        return {w:SOSpredictor().init(w, init_name, iteration) for w in t.wellnames_2}
    else:
        raise ValueError("Only SOS2 implemented!")
    
class SOSpredictor():
    def roundup(self, x):
        return int(math.ceil(x / 10.0))
    
    def rounddown(self, x):
        return int(math.floor(x / 10.0))
    
    def init(self, well, init_name, iteration):
        self.oil_vals, self.choke_vals = t.get_sos2_true_curves("oil", init_name, iteration)
        self.oil_vals = self.oil_vals[well]
        self.choke_vals = self.choke_vals[well]
        self.gas_vals, _ = t.get_sos2_true_curves("gas", init_name, iteration)
        self.gas_vals = self.gas_vals[well]
        
        self.p_type="sos2"
        return self
        
        
    def predict(self, choke, phase):
        index = None
        for i in range(len(self.choke_vals)):
            if(choke < round(self.choke_vals[i], 3)):
                index = i
                break
                
        factor = 1-(choke - self.choke_vals[index-1])/10
#        lower = self.rounddown(choke)
#        upper = self.roundup(choke)
#        factor = choke/10-lower
#        print("l:", index-1, "u:", index, "fac:", factor, "val_l:", self.gas_vals[index-1], "val_u:", self.gas_vals[index])
        if(phase=="oil"):
            return (factor)*self.oil_vals[index-1]+(1-factor)*self.oil_vals[index]
        else:
            return (factor)*self.gas_vals[index-1]+(1-factor)*self.gas_vals[index]