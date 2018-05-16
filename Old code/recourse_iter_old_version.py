# -*- coding: utf-8 -*-
"""
Created on Wed May 16 09:35:07 2018

@author: bendiw
"""

   
    def recourse_opt_old(self, init_name, max_changes, num_scen, pregen_scen=None, verbose=1, base_solution=None):
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
    