# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 22:49:52 2017

@author: Bendik
"""

from gurobipy import *
import MARS
import numpy as np
import tens
import math
#TODO: convert normalized values from MARS/NN to real oil/gas/gaslift/choke



m = Model("Ekofisk")

phasenames = ["oil", "gas"]
sepnames = ["LP", "HP"]
OIL = 0
GAS = 1
separator_dict = {"LP":2, "HP":1}

wellnames = ["A2", "A3", "A5", "A6", "A7", "A8", "B1", "B2", 
             "B3", "B4", "B5", "B6", "B7", "C1", "C2", "C3", "C4"]


well_to_sep = {"A2" : ["HP"], "A3": ["HP"], "A5": ["HP"], "A6": ["HP"], "A7": ["HP"], "A8": ["HP"], 
               "B1" : ["HP", "LP"], "B2" : ["HP", "LP"], "B3" : ["HP", "LP"], "B4" : ["HP", "LP"], "B5" : ["HP", "LP"], "B6" : ["HP", "LP"], "B7" : ["HP", "LP"], 
               "C1" : ["LP"], "C2" : ["LP"], "C3" : ["LP"], "C4" : ["LP"]}
platforms = ["A", "B", "C"]
p_dict = {"A" : ["A2", "A3", "A5", "A6", "A7", "A8"], "B":["B1", "B2", 
             "B3", "B4", "B5", "B6", "B7"], "C":["C1", "C2", "C3", "C4"]}
p_sep_route = {"A":[1], "B":[0,1], "C":[0]}
p_sep_names = {"A":["HP"], "B":["LP", "HP"], "C":["LP"]}
sep_p_route = {"LP": ["B", "C"], "HP":["A", "B"]}


#Case relevant numerics
sep_cap = {"LP": 5600000, "HP":math.inf}
tot_exp_cap = 5900000
glift_groups = [["A", "B"]]
glift_caps = [2000000]
max_changes = 21


oil_polytopes = {}
gas_polytopes = {}
well_oil_multidim = {}
well_gas_multidim = {}

multidims = [well_oil_multidim, well_gas_multidim]
polytopes = [oil_polytopes, gas_polytopes]
tens_prev = None
for platform in platforms:
    for well in p_dict[platform]:
#        if well in ["A8", "C1"]: #TODO: swap these with correct wells
#            continue
        for i in range(len(phasenames)):
            B_polys = {}
            B_multis = {}
            for separator in p_sep_names[platform]:
#                    print(separator_dict[separator])
<<<<<<< HEAD
                is_multi, polys = MARS.run(well, goal=phasenames[i], normalize=False, plot=False, hp=separator_dict[separator])
#                is_multi, polys = tens.run(well, goal=phasenames[i], normalize=False, plot=False, hp=separator_dict[separator])
=======
#                is_multi, polys = MARS.run(well, goal=phasenames[i], normalize=False, plot=False, hp=separator_dict[separator])
                is_multi, polys, tens_prev = tens.hey(well, prev=tens_prev, goal=phasenames[i], normalize=False, plot=False, hp=separator_dict[separator])
>>>>>>> master

                B_polys[separator] = polys
                B_multis[separator] = is_multi
            polytopes[i][well] = B_polys
            multidims[i][well] = B_multis
#        else:
#            for i in range(len(phasenames)):
#                is_multi, polys = MARS.run(well, phasenames[i], normalize=False, plot=False)
#               # is_multi, polys = tens.run(well, goal=phasenames[i])
#                polytopes[i][well] = polys
#                multidims[i][well] = is_multi




#phases = 1
constrPlatforms = [[]] #set of sets for platform gas lift supply constraints



#dictionaries to keep track of polytope variables
w_gas_polytope_vars = {}
w_oil_polytope_vars = {}
w_oil_breakpoint_vars = {well : {} for well in wellnames}
w_gas_breakpoint_vars = {well : {} for well in wellnames}

#dicts to hold change tracking variables:
w_change_vars = {well:[] for well in wellnames}

#dict with initial values for choke, gas lift per well, {well: [gas lift, choke]}
w_initial_vars = {well : [0,0] for well in wellnames}

#dict with binary var describing whether or not wells are producing in initial setting
w_initial_prod = {well : 0 for well in wellnames}

#dict with maximum gaslift in each well
w_max_glift = {"A2":124200.2899, "A3":99956.56739, "A5":125615.4024, "A6":150090.517, "A7":95499.28792, "A8":94387.68607, "B1":118244.94, "B2":112660.5625, "B3":138606.6016,
               "B4":197509.0709, "B5":210086.0959, "B6":117491.1591, "B7":125035.4286, "C1":106860.5264, "C2":132718.54, "C3" : 98934.12, "C4":124718.303}

#dict with allowed relative change, {well: [glift_delta, choke_delta]}
w_relative_change = {well : [1.0, 1.0] for well in wellnames}

#keep dicts in a list
w_polytope_vars = [w_oil_polytope_vars, w_gas_polytope_vars]
w_breakpoints = [w_oil_breakpoint_vars, w_gas_breakpoint_vars]

#w_prod_var = m.addVars(wellnames, name="w_prod_var", vtype=GRB.BINARY)

#WEIGHTING VARIABLES
for platform in platforms:
    for well in p_dict[platform]:
#        if(platform=="B"):
        for phase in range(len(phasenames)):
            already = 0
            #add list of polytope indicator variables for the well
            w_polytope_vars[phase][well] = {}
            
            #get corresponding polytope dict
            for separator in p_sep_names[platform]:
                ptopes = polytopes[phase]
                
                w_poly_temp = []
            
                if(multidims[phase][well][separator]):
                    #multidimensional case
                    brkpoints = {} #[] was list
        
                    for p in range(len(ptopes[well][separator])): #each polytope
                        #add polytope indicator variable
                        v = m.addVar(vtype = GRB.BINARY, name=well+"_"+phasenames[phase]+"_ptope_"+separator+"_"+str(p))
                        
                        #append to dictionary to keep track of polytope vars
                        w_poly_temp.append(v)
                        
                        #temp list to add polytope convexity constraint
                        for brk in ptopes[well][separator][p]: #each breakpoint
                            w = m.addVar(vtype = GRB.CONTINUOUS, name=well+"_"+phasenames[phase]+"_"+separator+"_"+str(p))
                            m.update()
                            brkpoints[w] = brk
                        
                        #convexity constraint on polytope
                        m.addConstr(quicksum(list(brkpoints.keys()))==v, well+"_"+phasenames[phase]+"_ptope_"+separator+"_"+str(p)+"_convex")
                    
                    #add breakpoints and polytope variables to dicts
                    w_breakpoints[phase][well][separator] = brkpoints
                    w_polytope_vars[phase][well][separator] = w_poly_temp
                else:
                    #single dimension case
                    brkpoints = {}
                    for p in range(len(ptopes[well])): #each breakpoint
                        w = m.addVar(vtype = GRB.CONTINUOUS, name=well+"_"+phasenames[phase]+"_"+str(p))
                        m.update()
                        brkpoints[w] = ptopes[well][separator][p]
                        
                    #SOS2 constraint on breakpoints
                    m.addSOS(2, list(brkpoints.keys()))
                    w_breakpoints[phase][well][separator] = brkpoints
            
print(w_polytope_vars[OIL]["A2"])
print(multidims[OIL]["A2"])
print(multidims[GAS]["A2"])


#PWL approximation variables/constraints
PWL_oil = {w : {} for w in wellnames}
PWL_gas = {w : {} for w in wellnames}
w_PWL = [PWL_oil, PWL_gas]
for platform in platforms:
    for well in p_dict[platform]:
        for phase in range(len(phasenames)):
            for separator in p_sep_names[platform]:
                
                #one approximation variable per well-separator pair (i.e. indexed by iu)
                pwlvar = m.addVar(vtype = GRB.CONTINUOUS, name=well+"_"+phasenames[phase]+"_PWL")
                w_PWL[phase][well][separator] = pwlvar
                
#                print(w_breakpoints[phase][well][separator].items())
#                print(multidims[phase])
                #enforce correct value
                if multidims[phase][well][separator]:
                    m.addConstr(pwlvar == quicksum([a*b[2] for a,b in w_breakpoints[phase][well][separator].items()]))
                    
                else:
                    m.addConstr(pwlvar == quicksum([a*b[1] for a,b in w_breakpoints[phase][well][separator].items()]))


        

            
#constraint to force PWL models to agree per well
for well in wellnames:
    for separator in well_to_sep[well]:
        if(multidims[OIL][well][separator]):
            m.addConstr(quicksum([a*b[0] for a,b in w_breakpoints[OIL][well][separator].items()]) - quicksum([a*b[0] for a,b in w_breakpoints[GAS][well][separator].items()])==0)
            m.addConstr(quicksum([a*b[1] for a,b in w_breakpoints[OIL][well][separator].items()]) - quicksum([a*b[1] for a,b in w_breakpoints[GAS][well][separator].items()])==0)
    #        pass
        else:
            m.addConstr(quicksum([a*b[0] for a,b in w_breakpoints[OIL][well][separator].items()]) - quicksum([a*b[0] for a,b in w_breakpoints[GAS][well][separator].items()])==0)

#big-M calc
#well_gas_max = {well: max([b[2] for a,b in w_breakpoints[GAS][well].items()]) if multidims[GAS][well] else max([b[1] for a,b in w_breakpoints[GAS][well].items()])  for well in wellnames}
#big_M= {}
#for pform in platforms:
#    for well in p_dict[pform]:
#        m_dict = {}
#        for sep in p_sep_route[pform]:
#            m_dict[sep] = min((well_gas_max[well]), sep_cap[sep])
#        big_M[well] = m_dict



w_route_flow_vars = {}
w_route_bin_vars = {}
#routing variables
for pform in platforms:
        for well in p_dict[pform]:
            well_sep_bin = {}#[] was list
            routes = []
            for sep in p_sep_names[pform]:
                routevar = m.addVar(vtype = GRB.BINARY, name=well+"_bin_route_"+sep+"_sep")
                well_sep_bin[sep] = routevar
                routes.append(routevar)
            m.addConstr(quicksum(routes)<=1, well+"_"+"routing_decision")
            w_route_bin_vars[well] = well_sep_bin
            
#single polytope selection constraints
for pform in platforms:
    for well in p_dict[pform]:
        for phase in range((len(phasenames))):
            for separator in p_sep_names[pform]:
#                print(pform, well, separator)
                if(multidims[phase][well][separator]):
#                    print(w_polytope_vars[phase][well])
#                    print(w_route_bin_vars[well])
                    m.addConstr(quicksum(w_polytope_vars[phase][well][separator])==w_route_bin_vars[well][separator])
                else:
                    m.addConstr(quicksum(list(w_breakpoints[phase][well][separator].keys()))==w_route_bin_vars[well][separator], well+"_"+phasenames[phase]+"_sos2_convex")


#separator constraints
for separator in sepnames:
    temp_wells = []
    for pform in sep_p_route[separator]:
        temp_wells.extend(p_dict[pform])
    if(separator=="LP"):
#        print(w_breakpoints[GAS][p_dict["C"][0]][separator])
        m.addConstr(quicksum([w_PWL[GAS][m][separator] for m in temp_wells]) - quicksum([quicksum([a*b[0] for a,b in w_breakpoints[GAS][m][separator].items()]) for m in p_dict["C"]]) <= sep_cap[separator], "LP_sep_gas_constr")                
    else:
        m.addConstr(quicksum([w_PWL[GAS][m][separator] for m in temp_wells]) <= sep_cap[separator], "HP_sep_gas_constr")

#gaslift constraints
for i in range(len(glift_groups)):
    m.addConstr(quicksum([quicksum([quicksum([a*b[0] for a,b in w_breakpoints[OIL][n][z].items()]) for n in p_dict["A"]]) for z in p_sep_names["A"]])
    + quicksum([quicksum([quicksum([a*b[0] for a,b in w_breakpoints[OIL][m][z].items()]) for m in p_dict["B"]]) for z in p_sep_names["B"]])<= glift_caps[i], "glift_a_b")
     
    
#total gas export constraint
#SUM of all gas produced in the entire field minus gas lift in all wells
m.addConstr(quicksum([quicksum([w_PWL[GAS][w][z] for z in well_to_sep[w]]) for w in wellnames]) 
- quicksum([quicksum([quicksum( [a*b[0] for a, b in w_breakpoints[GAS][w][z].items()]) for z in well_to_sep[w]]) for w in wellnames]) <= tot_exp_cap, "total_gas_export")



#change tracking variables
for well in wellnames:
    changevar = m.addVar(vtype = GRB.BINARY, name=well+"_glift_change_binary")
    w_change_vars[well].append(changevar)
    for separator in well_to_sep[well]:
#        print(well, multidims[OIL][well])
    #    for separator in well_to_sep[well]:
        m.addConstr(w_initial_vars[well][0] - quicksum([a*b[0] for a,b in w_breakpoints[OIL][well][separator].items()]) <=  changevar*w_initial_vars[well][0]*w_relative_change[well][0])
        m.addConstr(quicksum([a*b[0] for a,b in w_breakpoints[OIL][well][separator].items()]) - w_initial_vars[well][0] <=  changevar*w_initial_vars[well][0]*w_relative_change[well][0]+(1-w_initial_prod[well])*w_max_glift[well]*changevar)
        
for well in wellnames:
    if(any(multidims[OIL][well].values())):
        changevar_g = m.addVar(vtype = GRB.BINARY, name=well+"_choke_change_binary")
        w_change_vars[well].append(changevar_g)
        for separator in well_to_sep[well]:
            if(multidims[OIL][well][separator]):
                m.addConstr(w_initial_vars[well][1] - quicksum([a*b[1] for a,b in w_breakpoints[OIL][well][separator].items()]) <=  changevar_g*w_initial_vars[well][1]*w_relative_change[well][1])
                m.addConstr(quicksum([a*b[1] for a,b in w_breakpoints[OIL][well][separator].items()]) - w_initial_vars[well][1] <=  changevar_g*w_initial_vars[well][1]*w_relative_change[well][1]+ (1-w_initial_prod[well])*100*changevar_g)
        
    
    
#maximum changes constraint
m.addConstr(quicksum([quicksum(b) for a,b in w_change_vars.items()]) <= max_changes)

#objective vaue
m.setObjective(quicksum([quicksum([w_PWL[0][n][separator] for separator in well_to_sep[n]]) for n in wellnames]), GRB.MAXIMIZE)
m.update()
#m.setParam(GRB.Param.Heuristics, 0)
m.setParam(GRB.Param.Presolve, 0)
m.optimize()

vals = ["oil", "gas", "lift", "choke", "route"]
p_vals = ["oil", "gas", "lift", "changes"]
results = {well : {} for well in wellnames}
plat_res = {p : {} for p in platforms}

for p in platforms:
    p_tmp = {w:0 for w in p_vals}
    for well in p_dict[p]:
        tmp = {v: 0 for v in vals}
        tmp["route"] = "N/A"
        for separator in well_to_sep[well]:
    #        print(w_breakpoints[OIL]["A2"])
    #        print(well, separator)
    #        print(w_breakpoints[OIL])
#            print(well, m.getVarByName(well+"_glift_change_binary").x, (m.getVarByName(well+"_choke_change_binary").x if multidims[OIL][well][separator] else ""))
            w_change = m.getVarByName(well+"_glift_change_binary").x + (m.getVarByName(well+"_choke_change_binary").x if multidims[OIL][well][separator] else 0)
            tmp["oil"] += sum([a.x*b[2] for a,b in w_breakpoints[OIL][well][separator].items()])  if multidims[OIL][well][separator] else sum([a.x*b[1] for a,b in w_breakpoints[OIL][well][separator].items()])
            tmp["gas"] += sum([a.x*b[2] for a,b in w_breakpoints[GAS][well][separator].items()])  if multidims[GAS][well][separator] else sum([a.x*b[1] for a,b in w_breakpoints[GAS][well][separator].items()])
            tmp["lift"] += sum([a.x*b[0] for a,b in w_breakpoints[OIL][well][separator].items()])
            tmp["choke"] += sum([a.x*b[1] for a,b in w_breakpoints[OIL][well][separator].items()])  if multidims[OIL][well][separator] else 0
            if(m.getVarByName(well+"_bin_route_"+separator+"_sep").x == 1):
                tmp[vals[4]] = separator
        results[well] = tmp
        p_tmp["oil"] += tmp["oil"]
        p_tmp["gas"] += tmp["gas"]
        p_tmp["lift"] += tmp["lift"]
        p_tmp["changes"] += w_change
    plat_res[p] = p_tmp
    
sep_vals = {v : 0 for v in sepnames}
    
for sep in sepnames:
    sep_gas = 0
    for p in sep_p_route[sep]:
        for well in p_dict[p]:
            sep_gas += sum([a.x*b[2] for a,b in w_breakpoints[GAS][well][sep].items()])  if multidims[GAS][well][sep] else sum([a.x*b[1] for a,b in w_breakpoints[GAS][well][sep].items()]) - (sum([a.x*b[0] for a,b in w_breakpoints[OIL][well][separator].items()]) if sep=="LP" else 0)
    sep_vals[sep] = sep_gas

for p in platforms:
    print("Platform", p, "\t SUMS\tgas\t {0:8.2f} \toil:\t {1:8.2f} \tlift:\t {2:8.2f}".format(
            plat_res[p]["gas"], plat_res[p]["oil"], plat_res[p]["lift"]))
    print("Total changes:", plat_res[p]["changes"],  "\n")
    for well in p_dict[p]:
        print(well,"\troute:\t {0:.5} \tgas\t {1:8.2f} \toil:\t {2:8.2f} \tlift:\t {3:8.2f} \tchoke:\t{4:}".format(results[well]["route"],
              results[well]["gas"], results[well]["oil"], results[well]["lift"], ("N/A" if (results[well]["route"] != "N/A" and not multidims[OIL][well][results[well]["route"]]) else results[well]["choke"])))
    print("\n\n")
    
for sep in sepnames:
    print(sep, "Separator \ngas:\t", sep_vals[sep])
    print("slack:\t", m.getConstrByName(sep+"_sep_gas_constr").slack)
    print("RHS:\t", m.getConstrByName(sep+"_sep_gas_constr").RHS)
    print("\n")
#print("slack HP separator:", m.getConstrByName("HP_sep_gas_constr").slack)
print("Total gas export")
print("value:", sum(plat_res[p]["gas"]-plat_res[p]["lift"] for p in platforms))
print("slack:", m.getConstrByName("total_gas_export").slack, "\n")


print("Gas lift A & B")
print("value:", sum(plat_res[p]["lift"] for p in ["A", "B"]))
print("slack:", m.getConstrByName("glift_a_b").slack)

#print("gaslift A+B:", sum([sum([a.x*b[0] for a,b in w_breakpoints[OIL][n].items()]) for n in p_dict["A"]])+ sum([sum([a.x*b[0] for a,b in w_breakpoints[OIL][n].items()]) for n in p_dict["B"]]))