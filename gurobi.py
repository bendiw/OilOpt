# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 22:49:52 2017

@author: Bendik
"""

from gurobipy import *
import MARS
import numpy as np
import tens

#TODO: convert normalized values from MARS/NN to real oil/gas/gaslift/choke



m = Model("Ekofisk")

phasenames = ["oil", "gas"]
OIL = 0
GAS = 1
wellnames = ["A2", "A3", "A5", "A6", "A7", "A8", "B1", "B2", 
             "B3", "B4", "B5", "B6", "B7", "C1", "C2", "C3", "C4"]
platforms = ["A", "B", "C"]
p_dict = {"A" : ["A2", "A3", "A5", "A6", "A7", "A8"], "B":["B1", "B2", 
             "B3", "B4", "B5", "B6", "B7"], "C":["C1", "C2", "C3", "C4"]}
p_sep_route = {"A":[1], "B":[0,1], "C":[0]}
sep_p_route = [["B"], ["A", "B"]]

sep_cap = [1, math.inf]
tot_exp_cap = 10


oil_polytopes = {}
gas_polytopes = {}
well_oil_multidim = {}
well_gas_multidim = {}

multidims = [well_oil_multidim, well_gas_multidim]
polytopes = [oil_polytopes, gas_polytopes]
for well in wellnames:
    print(well)
    for i in range(len(phasenames)):
        is_multi, polys = MARS.run(well, phasenames[i], plot=False)
#        is_multi, polys = tens.run(well, goal=phasenames[i])
        polytopes[i][well] = polys
        multidims[i][well] = is_multi





#phases = 1
constrPlatforms = [[]] #set of sets



#dictionaries to keep track of polytope variables
w_gas_polytope_vars = {}
w_oil_polytope_vars = {}
w_oil_breakpoint_vars = {well : {} for well in wellnames}
w_gas_breakpoint_vars = {well : {} for well in wellnames}


#keep dicts in a list
w_polytopes = [w_oil_polytope_vars, w_gas_polytope_vars]
w_breakpoints = [w_oil_breakpoint_vars, w_gas_breakpoint_vars]



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
            m.addConstr(quicksum(list(brkpoints.keys()))==1, well+"_"+phasenames[phase]+"_sos2_convex")
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


        
#single polytope selection constraints
for well in wellnames:
    for phase in range((len(phasenames))):
        if(multidims[phase][well]):
            m.addConstr(quicksum(w_polytopes[phase][well])<=1)
            
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

#routing variables
for pform in platforms:
    if(pform=="C"): #local separator
        routerate_C = m.addVar(vtype = GRB.CONTINUOUS, name="C_"+"flow_local_sep")
        m.addConstr(routerate_C + quicksum([a*b[0] for a,b in w_breakpoints[OIL][well].items()]) - w_PWL[1][well] == 0, well+"_local_gasflow_tracking")
    else:
        for well in p_dict[pform]:
            well_sep_rate = {}
            routes = []
            rates = []
            for sep in p_sep_route[pform]:
                routevar = m.addVar(vtype = GRB.BINARY, name=well+"_"+"bin_route_sep_"+str(sep))
                routerate = m.addVar(vtype = GRB.CONTINUOUS, name=well+"_"+"flow_sep_"+str(sep))
                well_sep_rate[sep] = routerate
                #big-M constraint on flow
                m.addConstr(routerate - routevar*big_M[well][sep]<=0, well+"_"+"routing_decision")

                routes.append(routevar)
                rates.append(routerate)
            m.addConstr(quicksum(routes)<=1, well+"_"+"routing_decision")
            m.addConstr(quicksum(rates)-w_PWL[1][well] == 0, well+"_gasflow_tracking")
            w_route_flow_vars[well] = well_sep_rate

#separator constraints
for sep in range(len(sep_cap)):
    if(sep==0):
        m.addConstr(quicksum(w_route_flow_vars[well][sep] for well in p_dict[sep_p_route[sep][0]]) + routerate_C <= sep_cap[sep], "sep_"+str(sep)+"_gas_constr")
    else:
        m.addConstr(quicksum(w_route_flow_vars[well][sep] for well in p_dict[sep_p_route[sep][0]]) + quicksum(w_route_flow_vars[well][sep] for well in p_dict[sep_p_route[sep][1]]) <= sep_cap[sep], "sep_"+str(sep)+"_gas_constr")

        for pform in sep_p_route[sep]:
            for well in p_dict[pform]:
                pass
#change tracking variables


#CONSTRAINT CREATION

#routing constraints

#maximum changes constraint

#separator constraints

#gaslift constraints


#EXAMPLE
#x = m.addVar(vtype= GRB.CONTINUOUS)
#y = m.addVar(vtype= GRB.CONTINUOUS)
#z = m.addVar(vtype= GRB.CONTINUOUS)

m.setObjective(quicksum([w_PWL[0][well] for well in wellnames]), GRB.MAXIMIZE)
# Add constraint: x + 2 y + 3 z <= 4
#m.addConstr(b == quicksum([a*b for a,b in zip(var, gas)]))
#m.addConstr(b <=2.5, "c0")
#for v in var:
#    m.addConstr(v<=1)

# Add constraint: x + y >= 1
m.optimize()
for v in m.getVars():
    print(v.varName, v.x)


for c in m.getConstrs():    
    print("constr", c.ConstrName, "slack ", c.slack)
print('Obj:', m.objVal)