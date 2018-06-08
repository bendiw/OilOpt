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
import time
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
def recourse(num_iter=200, num_scen=10, max_changes=3, init_name=None, model_type="sos2", 
             verbose=0, save=False, from_infeasible=False, simul_change=False, undo_allow_on=False, 
             perfect_info=False, start_iter=0, indiv_caps=True):
    
    filestring = "results/robust_recourse_iterative/"+init_name+"/"+model_type+"/"+str(num_iter)+"iter_"+str(num_scen)+"scen_"+init_name+"_"+model_type+("_simul" if simul_change else"")+("_EVPI" if perfect_info else "")+".csv"
    if perfect_info:
        simul_change=True
        num_scen=1
    #init model
    model = get_model(model_type).init(max_changes=max_changes, num_scen=num_scen, init_name=init_name, indiv_caps=indiv_caps) #, w_relative_change={w:[0.5] for w in t.wellnames_2}
    if(undo_allow_on and "over_cap" in init_name):
        model.undo_allow_on_off()
        print("wells may be switched on/off.")
    
    #failsafe check in case we forget to specify :)
    if(init_name=="over_cap" or init_name=="over_cap_old"):
        from_infeasible=True
        
        
    #if we have perfect information there is no need to perform changes sequentially
    #also only require a single scenario in this case
        
    #this is our starting point in terms of chokes
    init_chokes = t.get_init_chokes(init_name)
    if(verbose>0):
        print("Recourse from initial scenario", "'"+init_name+"'")
        print("Initial choke settings:")
        for w in t.wellnames_2:
            print(w, "\t",init_chokes[w])
    
    #store results in a dataframe
    results = pd.DataFrame(columns=t.recourse_algo_columns)
    
    #get first solution since it will be same for all iterations unless we are in PI/WS model
    if not perfect_info:
        model.solve(verbose=max(verbose-2, 0))
        first_sol = model.get_chokes()
    
    
    for i in range(start_iter,num_iter):
        stdout.write("\r iteration %d" % i)
        stdout.flush()
#        print(model.exp_constr[0].getAttr("rhs"))
#        print(model.get_solution())
        true_well_curves = get_true_models(init_name, i)
        if perfect_info: #now the first solution depends on the scenario we are in.
            model.m.update() #need to do this in order to swap well curves
            for w in t.wellnames_2:
#                print("true", w)
                model.set_true_curve(w, true_well_curves[w], perfect_info=perfect_info)

            model.solve(verbose=max(verbose-2, 0))
            first_sol = model.get_chokes()
    
        #get "true" sos2 or NN models
        #iterate with one less change since we have already found first solution
        if(not simul_change):
            results.loc[i] = iteration(model, init_chokes, first_sol, max_changes-1, true_well_curves, verbose=verbose, from_infeasible=from_infeasible, init_name=init_name, perfect_info=perfect_info)
        else:
            results.loc[i] = simul_iteration(model, init_chokes, first_sol, true_well_curves, verbose=verbose)
        #revert to initial model
        if not perfect_info:
            model.set_chokes(first_sol)
        model.reset_m()
        model.redo_allow_on_off()
        model.m.update()
#        print(model.oil_out_constr)
    if(save):
        results.to_csv(filestring, sep=';')
    print("\n")
    return results

# =============================================================================
# Perform one iteration of algo with simultaneous changes.
# =============================================================================
def simul_iteration(model, init_chokes, first_sol, true_well_curves, verbose=0):
    changed_w = []
    ch_gas = {}
    ch_oil = {}
    old_gas ={}
    old_oil = {}
    tot_oil = sum([true_well_curves[w].predict(init_chokes[w], "oil") for w in t.wellnames_2])
    tot_gas = sum([true_well_curves[w].predict(init_chokes[w], "gas") for w in t.wellnames_2])
    infeasible_count = 0
    for w in t.wellnames_2:
        if(abs(first_sol[w]-init_chokes[w]) >= 0.01):
            changed_w.append(w)
    
        
    for w in changed_w:
        old_gas[w] = true_well_curves[w].predict(init_chokes[w], "gas")
        ch_gas[w] = true_well_curves[w].predict(first_sol[w], "gas")
        old_oil[w] = true_well_curves[w].predict(init_chokes[w], "oil")
        ch_oil[w] = true_well_curves[w].predict(first_sol[w], "oil")
        if(ch_gas[w]-0.01 > model.well_cap[w]):
            infeasible_count+=1
    new_tot_gas = tot_gas-sum(old_gas.values())+sum(ch_gas.values())
    new_tot_oil = tot_oil-sum(old_oil.values())+sum(ch_oil.values())
    
    if(new_tot_gas-0.01 > model.tot_exp_cap):
        infeasible_count+=1
    if(verbose>1):
        print("\n==================================================")
        print("Solution was", ("infeasible!" if infeasible_count>0 else "feasible."))
        print("\nchanged wells:")
        for w in changed_w:
            print(w+"\t[", (round(init_chokes[w],2)),"--->", round(first_sol[w],2), "]")
        print("\ntot_oil:\t", round(new_tot_oil,2), "\t[", round(tot_oil,2), "]")
        print("tot_gas:\t", round(new_tot_gas,2), "\t[", round(tot_gas,2), "]")
        print("\n==================================================\n")
    rowlist = [infeasible_count, new_tot_oil, new_tot_gas]+ list(first_sol.values())
    return rowlist


# =============================================================================
# Perform one iteration of algorithm. Step-wise changes
# =============================================================================
def iteration(model, init_chokes, first_sol, changes, true_well_curves, verbose=0, from_infeasible=False, init_name=None, perfect_info=False):
    #store implemented chokes, first we use only the starting point
    impl_chokes = init_chokes
    infeasible_count = 0
    tot_oil = sum([true_well_curves[w].predict(init_chokes[w], "oil") for w in t.wellnames_2])
    tot_gas = sum([true_well_curves[w].predict(init_chokes[w], "gas") for w in t.wellnames_2])
#    print("initchoke", init_chokes, "oil", tot_oil, "\ngas", tot_gas)
    new_chokes = first_sol
    
    #since we start by checking for #changes+1, we need to iterate an 'extra' time including 0
    for c in range(changes, -1, -1):
        suggested_tot_oil_all_changes = sum([true_well_curves[w].predict(new_chokes[w], "oil") for w in t.wellnames_2])
        suggested_tot_gas_all_changes = sum([true_well_curves[w].predict(new_chokes[w], "gas") for w in t.wellnames_2])
        inf_single, tot_oil, tot_gas, impl_chokes, change_well, change_gas = check_and_impl_change(true_well_curves, tot_oil, tot_gas, impl_chokes, new_chokes, model.well_cap, model.tot_exp_cap, from_infeasible)
        


        infeasible_count+=inf_single 
        
        if inf_single<1:
            from_infeasible=False
        
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
                
            print("\ntot oil:\t", tot_oil,"\tsugg. tot oil after all changes:\t", suggested_tot_oil_all_changes, "\ntot gas:\t", tot_gas, "\tsugg. tot gas after all changes:\t", suggested_tot_gas_all_changes)
            print("==================================================\n")
            
            
#        print("res\n", opt_results)
        old_chokes = {key: value for key, value in impl_chokes.items()}
        model.set_chokes(impl_chokes)
        
        model.set_changes(c)
        if(c==1):
            model.undo_allow_on_off()

        #if we came from infeasibility, only care about whether or not we ended up in infeasibility
        if(c<1):
            if(from_infeasible):
                if(inf_single==1):
                    infeasible_count=1
                else:
                    infeasible_count=0
            elif init_name == "over_cap":
                infeasible_count=0
            break
        if not perfect_info:
            #allow individual infeasibility as a quick fix for single wells
            if(init_name=="over_cap" and change_gas > model.well_cap[change_well]):
    #            print("\n***\n"+change_well,"indiv cap changed to",change_gas,"\n***")
                model.set_indiv_gas(change_gas, change_well)
    #        model.set_changes(changes)
            if(init_name == "over_cap" or init_name=="over_cap_old"):
                if(impl_chokes[change_well] != 0):
                    model.set_true_curve(change_well, true_well_curves[change_well])
            else:
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
# implement a change if feasible, or if we came from infeasibility we perform
# the change anyway.
# =============================================================================
def check_and_impl_change(true_well_curves, tot_oil, tot_gas, old_chokes, new_chokes, indiv_cap, tot_cap, from_infeasible):
#    tot_gas = 0
    found1=False
    found2=False
    #find which well to change. prioritize negative changes
    change_well1 = None
    change_well2 = None
    for w in reversed(t.well_order):
        if(new_chokes[w] - old_chokes[w] < -0.01):
            if(new_chokes[w]<0.01 and not found2):
                change_well2 = w
                found2=True
            if(new_chokes[w]>0.01):
                change_well1 = w
                found1=True
                break
    if not (found1 or found2):
        for w in t.well_order:
            if(abs(new_chokes[w]-old_chokes[w]) >= 0.01):
                change_well1 = w
                found1=True
                break
    if not (found1 or found2):
        return 0, tot_oil, tot_gas, old_chokes, None, None
    if(found1):
        change_well = change_well1
    else:
        change_well = change_well2
    #implement change and check for infeasibility
    temp_chokes = {key: value for key, value in old_chokes.items()}
    temp_chokes[change_well] = new_chokes[change_well]
#    print("\nchangewell:", change_well)
    old_gas = true_well_curves[change_well].predict(old_chokes[change_well], "gas")
    change_gas = true_well_curves[change_well].predict(temp_chokes[change_well], "gas")
    old_oil = true_well_curves[change_well].predict(old_chokes[change_well], "oil")
    change_oil = true_well_curves[change_well].predict(temp_chokes[change_well], "oil")
    if(change_gas-0.01 > indiv_cap[change_well] or tot_gas+(change_gas-old_gas)-0.01 > tot_cap):
#        if(change_gas-0.01 > indiv_cap[change_well]):
#            print("\nINDIV INFEASIBLE:", change_well, "\t",change_gas, "\n")
        if(from_infeasible):
            #implement the change
            return 1, tot_oil+(change_oil-old_oil), tot_gas+(change_gas-old_gas), temp_chokes, change_well, change_gas
        #else, revert
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
#         Calculates the switch-off penalty from stored results of "over_cap" runs
# =============================================================================
def switch_off_penalty(scen, init_name="over_cap", model_type="sos2", simul_changes=False, verbose=0, save=False):
    df = pd.DataFrame(columns=["switch-off penalty"])
    c_cols = [w+"_choke_final" for w in t.wellnames_2]
    folder = "results/robust_recourse_iterative/"+init_name+"/"+(model_type if not simul_changes else "simul")+"/"
    filename = "200iter_"+str(scen)+"scen_"+init_name+"_"+model_type+("_simul" if simul_changes else "")
    off_order = list(reversed(t.well_order))
    results = pd.read_csv(folder+filename+".csv", sep=';', index_col=0)
    for i in range(results.shape[0]):
        i_res = results.loc[i]
        if(i_res["infeasible count"]>0):
            true_well_curves = get_true_models(init_name, i)
            chks = i_res[c_cols]
            gas = {w:true_well_curves[w].predict(chks[w+"_choke_final"], "gas") for w in t.wellnames_2}
            oil = {w:true_well_curves[w].predict(chks[w+"_choke_final"], "oil") for w in t.wellnames_2}
            off_wells = []
            for k, v in gas.items():
                if(v-0.01 > 54166):
                    off_wells.append(k)
            tot_oil = sum([oil[w] if w not in off_wells else 0 for w in t.wellnames_2])
            tot_gas = sum([gas[w] if w not in off_wells else 0 for w in t.wellnames_2])
            while(tot_gas-0.01 > t.tot_exp_caps["over_cap"]):
                if(off_order[len(off_wells)] not in off_wells):
                    off_wells.append(off_order[len(off_wells)])
                    tot_gas = sum([gas[w] if w not in off_wells else 0 for w in t.wellnames_2])
                    tot_oil = sum([oil[w] if w not in off_wells else 0 for w in t.wellnames_2])

            df.loc[i] = tot_oil
            if(verbose>0):
                print("\niter:", i, "off:", off_wells,"oils:", [oil[w] for w in off_wells], "gases:", [gas[w] for w in off_wells],"oil:", tot_oil, "gas:", tot_gas)
        else:
            df.loc[i] = i_res["oil output"]
    if(save):
        df.to_csv(folder+filename+"off_penalty"+".csv", sep=";")
    return df
        

# =============================================================================
# Function to time runs
# =============================================================================
def time_test(init_name, max_changes=3, model_type="sos2", num_tests=10, save=False, verbose=0, start_scen=5, max_scen=500):
    scens_list = [ 5,	10,	15,	20,	25,	30,	40,	50,75,100,125,150,200,300,400,500]
    scens = []
    for s in scens_list:
        if s >= start_scen and s<=max_scen:
            scens.append(s)
    z = np.zeros((len(scens), 13))
    columns = ["scenarios", "mean time", "std time"] + [i for i in range(10)]
    for s in scens:
        if(s > max_scen):
            break
        t = []
        for n in range(num_tests):
            model = get_model(model_type).init(init_name=init_name, num_scen=s, max_changes=max_changes, stability_iter=n)
            start = time.time()
            model.solve(0)
            elapsed = time.time()-start
            t.append(elapsed)
        rowlist = [s, np.mean(t), np.std(t)]+t
        if(verbose>0):
            print("\nscenarios:", s, "mean time:", round(rowlist[1],5), "std time:", round(rowlist[2], 5))
        z[scens.index(s)] = rowlist
    df = pd.DataFrame(z, columns=columns)
    if save:
        df.to_csv("results/technical/"+model_type+"_"+init_name+"_"+str(num_tests)+"_tests.csv",sep=";")
    return df
        
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
        if(choke<=0.001):
            return 0.
        for i in range(len(self.choke_vals)):
            if(choke < round(self.choke_vals[i], 3)):
                index = i
                break
                
        factor = 1-(choke - self.choke_vals[index-1])/(self.choke_vals[index]-self.choke_vals[index-1])
        if(phase=="oil"):
            return (factor)*self.oil_vals[index-1]+(1-factor)*self.oil_vals[index]
        else:
            return (factor)*self.gas_vals[index-1]+(1-factor)*self.gas_vals[index]