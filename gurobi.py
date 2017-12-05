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
wellnames = ["A2", "A3", "A5", "A6", "A7", "A8", "B1", "B2", 
             "B3", "B4", "B5", "B6", "B7", "C1", "C2", "C3", "C4"]
platforms = ["A", "B", "C"]
p_dict = {"A" : ["A2", "A3", "A5", "A6", "A7", "A8"], "B":["B1", "B2", 
             "B3", "B4", "B5", "B6", "B7"], "C":["C1", "C2", "C3", "C4"]}
p_sep_route = {"A":[1], "B":[0,1], "C":[0]}
sep_p_route = [["B"], ["A", "B"]]


#Case relevant numerics
sep_cap = [100000, math.inf]
tot_exp_cap = 510000
glift_groups = [["A", "B"]]
glift_caps = [500000]
max_changes = 100


oil_polytopes = {}
gas_polytopes = {}
well_oil_multidim = {}
well_gas_multidim = {}

multidims = [well_oil_multidim, well_gas_multidim]
polytopes = [oil_polytopes, gas_polytopes]
for well in wellnames:
    print(well)
    for i in range(len(phasenames)):
        is_multi, polys = MARS.run(well, phasenames[i], normalize=False, plot=False)
       # is_multi, polys = tens.run(well, goal=phasenames[i])
        polytopes[i][well] = polys
        multidims[i][well] = is_multi




#phases = 1
constrPlatforms = [[]] #set of sets



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
w_polytopes = [w_oil_polytope_vars, w_gas_polytope_vars]
w_breakpoints = [w_oil_breakpoint_vars, w_gas_breakpoint_vars]

#w_prod_var = m.addVars(wellnames, name="w_prod_var", vtype=GRB.BINARY)

#WEIGHTING VARIABLES
for well in wellnames:
    for phase in range(len(phasenames)):
        already = 0
        #add list of polytope indicator variables for the well
        w_polytopes[phase][well] = []
        
        #get corresponding polytope dict
        ptopes = polytopes[phase]
        if(multidims[phase][well]):
            #multidimensional case
            brkpoints = {}

            for p in range(len(ptopes[well])): #each polytope
                
                #add polytope indicator variable
                v = m.addVar(vtype = GRB.BINARY, name=well+"_"+phasenames[phase]+"_polytope_"+str(p))
                
                #append to dictionary to keep track of polytope vars
                w_polytopes[phase][well].append(v)
                
                #temp list to add polytope convexity constraint
                for brk in ptopes[well][p]: #each breakpoint
                    w = m.addVar(vtype = GRB.CONTINUOUS, name=well+"_"+phasenames[phase]+"_"+str(p))
                    m.update()
                    brkpoints[w] = brk
                
                #convexity constraint on polytope
                m.addConstr(quicksum(list(brkpoints.keys()))==v, well+"_"+phasenames[phase]+"_ptope_"+str(p)+"_convex")
            w_breakpoints[phase][well] = brkpoints
        else:
            #single dimension case
            brkpoints = {}
            for p in range(len(ptopes[well])): #each breakpoint
                w = m.addVar(vtype = GRB.CONTINUOUS, name=well+"_"+phasenames[phase]+"_"+str(p))
                m.update()
                brkpoints[w] = ptopes[well][p]
                
#            print(well, phasenames[phase])
#            print(brkpoints)
#            print("\n\n")
            #SOS2 constraint on breakpoints
            m.addSOS(2, list(brkpoints.keys()))
            w_breakpoints[phase][well] = brkpoints
            

#PWL approximation variables/constraints
PWL_oil = {}
PWL_gas = {}
w_PWL = [PWL_oil, PWL_gas]
for well in wellnames:
    for phase in range(len(phasenames)):
        pwlvar = m.addVar(vtype = GRB.CONTINUOUS, name=well+"_"+phasenames[phase]+"_PWL")
        w_PWL[phase][well] = pwlvar
        if multidims[phase][well]:
            m.addConstr(pwlvar == quicksum([a*b[2] for a,b in w_breakpoints[phase][well].items()]))
            
        else:
            m.addConstr(pwlvar == quicksum([a*b[1] for a,b in w_breakpoints[phase][well].items()]))


        

            
#constraint to force PWL models to agree per well
for well in wellnames:
    if(multidims[OIL][well]):
        m.addConstr(quicksum([a*b[0] for a,b in w_breakpoints[OIL][well].items()]) - quicksum([a*b[0] for a,b in w_breakpoints[GAS][well].items()])==0)
        m.addConstr(quicksum([a*b[1] for a,b in w_breakpoints[OIL][well].items()]) - quicksum([a*b[1] for a,b in w_breakpoints[GAS][well].items()])==0)
#        pass
    else:
        m.addConstr(quicksum([a*b[0] for a,b in w_breakpoints[OIL][well].items()]) - quicksum([a*b[0] for a,b in w_breakpoints[GAS][well].items()])==0)

#big-M calc
well_gas_max = {well: max([b[2] for a,b in w_breakpoints[GAS][well].items()]) if multidims[GAS][well] else max([b[1] for a,b in w_breakpoints[GAS][well].items()])  for well in wellnames}
big_M= {}
for pform in platforms:
    for well in p_dict[pform]:
        m_dict = {}
        for sep in p_sep_route[pform]:
            m_dict[sep] = min((well_gas_max[well]), sep_cap[sep])
        big_M[well] = m_dict



w_route_flow_vars = {}
w_route_bin_vars = {}
#routing variables
for pform in platforms:
    if(pform=="C"): #local separator
        for well in p_dict[pform]:
            routevar = m.addVar(vtype = GRB.BINARY, name=well+"_"+"produce_bin")
            w_route_bin_vars[well] = [routevar]

        routerate_C = m.addVar(vtype = GRB.CONTINUOUS, name="C_"+"flow_local_sep")
        m.addConstr(routerate_C + quicksum([quicksum([a*b[0] for a,b in w_breakpoints[OIL][n].items()]) for n in p_dict[pform]]) - quicksum(w_PWL[1][n] for n in p_dict[pform]) == 0, pform+"_local_gasflow_tracking")
    else:
        for well in p_dict[pform]:
            well_sep_rate = {}
            well_sep_bin = []
            routes = []
            rates = []
            for sep in p_sep_route[pform]:
                routevar = m.addVar(vtype = GRB.BINARY, name=well+"_bin_route_"+sepnames[sep]+"_sep")
                routerate = m.addVar(vtype = GRB.CONTINUOUS, name=well+"_"+"flow_"+sepnames[sep]+"_sep")
                well_sep_rate[sep] = routerate
                well_sep_bin.append(routevar)
                #big-M constraint on flow
                m.addConstr(routerate - routevar*big_M[well][sep]<=0, well+"_"+"routing_bigM")

                routes.append(routevar)
                rates.append(routerate)
            m.addConstr(quicksum(routes)<=1, well+"_"+"routing_decision")
            m.addConstr(quicksum(rates)-w_PWL[1][well] == 0, well+"_gasflow_tracking")
            w_route_flow_vars[well] = well_sep_rate
            w_route_bin_vars[well] = well_sep_bin
            
#single polytope selection constraints
for pform in platforms:
    for well in p_dict[pform]:
        for phase in range((len(phasenames))):
            if(multidims[phase][well]):
                m.addConstr(quicksum(w_polytopes[phase][well])==quicksum(w_route_bin_vars[well]))
            else:
                m.addConstr(quicksum(list(w_breakpoints[phase][well].keys()))==quicksum(w_route_bin_vars[well]), well+"_"+phasenames[phase]+"_sos2_convex")


#separator constraints
for sep in range(len(sep_cap)):
    if(sep==0):
        m.addConstr(quicksum(w_route_flow_vars[n][sep] for n in p_dict[sep_p_route[sep][0]]) + routerate_C <= sep_cap[sep], "LP_sep_gas_constr")
    else:
        m.addConstr(quicksum(w_route_flow_vars[n][sep] for n in p_dict[sep_p_route[sep][0]]) + quicksum(w_route_flow_vars[n][sep] for n in p_dict[sep_p_route[sep][1]]) <= sep_cap[sep], "HP_sep_gas_constr")

#gaslift constraints
for i in range(len(glift_groups)):
    m.addConstr(quicksum([quicksum([a*b[0] for a,b in w_breakpoints[OIL][n].items()]) for n in p_dict["A"]])
    + quicksum([quicksum([a*b[0] for a,b in w_breakpoints[OIL][m].items()]) for m in p_dict["B"]])<= glift_caps[i], "glift_a_b")
     
    
#total gas export constraint
#flow from separators minus gas lift employed in fields A + B
m.addConstr(quicksum(w_route_flow_vars[n][0] for n in p_dict[sep_p_route[0][0]]) + 
            routerate_C +
            quicksum([quicksum(w_route_flow_vars[n][1] for n in p_dict[sep_p_route[1][j]]) for j in range(len(sep_p_route[1]))])
            - quicksum([quicksum([quicksum([a*b[0] for a,b in w_breakpoints[OIL][n].items()]) for n in p_dict[pformz]]) for pformz in glift_groups[0]])
            <= tot_exp_cap, "total_gas_export")


#change tracking variables
for well in wellnames:
    changevar = m.addVar(vtype = GRB.BINARY, name=well+"_glift_change_binary")
    w_change_vars[well].append(changevar)
    m.addConstr(w_initial_vars[well][0] - quicksum([a*b[0] for a,b in w_breakpoints[OIL][well].items()]) <=  changevar*w_initial_vars[well][0]*w_relative_change[well][0])
    m.addConstr(quicksum([a*b[0] for a,b in w_breakpoints[OIL][well].items()]) - w_initial_vars[well][0] <=  changevar*w_initial_vars[well][0]*w_relative_change[well][0]+(1-w_initial_prod[well])*w_max_glift[well]*changevar)
    
    if(multidims[OIL][well]):
        changevar = m.addVar(vtype = GRB.BINARY, name=well+"_choke_change_binary")
        w_change_vars[well].append(changevar)
        m.addConstr(w_initial_vars[well][1] - quicksum([a*b[1] for a,b in w_breakpoints[OIL][well].items()]) <=  changevar*w_initial_vars[well][1]*w_relative_change[well][1])
        m.addConstr(quicksum([a*b[1] for a,b in w_breakpoints[OIL][well].items()]) - w_initial_vars[well][1] <=  changevar*w_initial_vars[well][1]*w_relative_change[well][1]+ (1-w_initial_prod[well])*changevar)
    
#maximum changes constraint
m.addConstr(quicksum([quicksum(b) for a,b in w_change_vars.items()]) <= max_changes)

#objective vaue
m.setObjective(quicksum([w_PWL[0][n] for n in wellnames]), GRB.MAXIMIZE)
m.update()
m.optimize()


print('\nObj value:', m.objVal, "\n")
for p in platforms:
    print(p,"gas lift", sum([sum([a.x*b[0] for a,b in w_breakpoints[OIL][q].items()]) for q in p_dict[p]]))
    for well in p_dict[p]:
        print(well,"\tHP_prod?\t {0:2.1f} \tgas\t {1:8.2f} \toil:\t {2:8.2f} \tlift:\t {3:8.2f}".format(m.getVarByName(well+"_"+"bin_route_HP_sep").x if p!="C" else m.getVarByName(well+"_"+"produce_bin").x, 
     sum([a.x*b[2] for a,b in w_breakpoints[GAS][well].items()]) if multidims[GAS][well] else sum([a.x*b[1] for a,b in w_breakpoints[GAS][well].items()]),
     sum([a.x*b[2] for a,b in w_breakpoints[OIL][well].items()]) if multidims[OIL][well] else sum([a.x*b[1] for a,b in w_breakpoints[OIL][well].items()]),
     sum([a.x*b[0] for a,b in w_breakpoints[OIL][well].items()])) )
    print("\n")
    
    

print("slack LP separator:", m.getConstrByName("LP_sep_gas_constr").slack)
print("slack HP separator:", m.getConstrByName("HP_sep_gas_constr").slack)
print("slack tot gas exp:", m.getConstrByName("total_gas_export").slack)
print("gaslift A+B:", sum([sum([a.x*b[0] for a,b in w_breakpoints[OIL][n].items()]) for n in p_dict["A"]])+ sum([sum([a.x*b[0] for a,b in w_breakpoints[OIL][n].items()]) for n in p_dict["B"]]))