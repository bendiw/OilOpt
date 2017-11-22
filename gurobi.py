# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 22:49:52 2017

@author: Bendik
"""

from gurobipy import *
import MARS
import numpy as np

#TODO: convert normalized values from MARS/NN to real oil/gas/gaslift/choke



m = Model("Ekofisk")

phasenames = ["oil", "gas"]
wellnames = ["A2", "A3", "A5", "A6", "A7", "A8", "B1", "B2", 
             "B3", "B4", "B5", "B6", "B7", "C1", "C2", "C3", "C4"]

oil_polytopes = {}
gas_polytopes = {}
well_oil_multidim = {}
well_gas_multidim = {}

multidims = [well_oil_multidim, well_gas_multidim]
polytopes = [oil_polytopes, gas_polytopes]
for well in wellnames:
    print(well)
    for i in range(len(phasenames)):
        is_multi, polys = MARS.run(well, phasenames[i])
        polytopes[i][well] = polys
        multidims[i][well] = is_multi

#nn = tens()




wells = []
phases = 1
platforms = []
separators = []
constrPlatforms = [[]] #set of sets



#dictionaries to keep track of polytope variables
w_gas_polytope_vars = {}
w_oil_polytope_vars = {}
w_oil_breakpoint_vars = {well : [] for well in wellnames}
w_gas_breakpoint_vars = {well : [] for well in wellnames}

#keep dicts in a list
w_polytopes = [w_oil_polytope_vars, w_gas_polytope_vars]
w_breakpoints = [w_oil_breakpoint_vars, w_gas_breakpoint_vars]

#VARIABLE CREATION
#b = m.addVar(vtype= GRB.CONTINUOUS, name="totgas")

#WEIGHTING VARIABLES
for well in wellnames:
    
    for phase in range(len(phasenames)):
        
        #add list of polytope indicator variables for the well
        w_polytopes[phase][well] = []
        
        #get corresponding polytope dict
        ptopes = polytopes[phase]

        if(multidims[phase][well]):
            #multidimensional case
            for p in range(len(ptopes[well])): #each polytope
                
                #add polytope indicator variable
                v = m.addVar(vtype = GRB.BINARY, name=well+"_"+phasenames[phase]+"_polytope_"+str(p))
                
                #append to dictionary to keep track of var
                w_polytopes[phase][well].append(v)
                
                #temp list to add polytope convexity constraint
                brkpoints = []
                for brk in ptopes[well][p]: #each breakpoint
                    w = m.addVar(vtype = GRB.CONTINUOUS, name=well+"_"+phasenames[phase]+"_"+str(p))
                    brkpoints.append(w)
                
                #convexity constraint on polytope
                m.addConstr(quicksum(brkpoints)==v, well+"_"+phasenames[phase]+"_ptope_"+str(p)+"_convex")
                w_breakpoints[phase][well].append(brkpoints)
        else:
            #single dimension case
            brkpoints = []
            for brk in ptopes[well]: #each breakpoint
                    w = m.addVar(vtype = GRB.CONTINUOUS, name=well+"_"+phasenames[phase]+"_"+str(p))
                    brkpoints.append(w)
                
            #SOS2 constraint on breakpoints
            m.addSOS(2, brkpoints)
            m.addConstr(quicksum(brkpoints)==1, well+"_"+phasenames[phase]+"_ptope_sos2_convex")
            w_breakpoints[phase][well] = brkpoints
            
m.update() #test
for c in m.getConstrs():    
    print("constr", c.ConstrName)

#PWL approximation variables

#routing variables

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

m.setObjective(quicksum([a*b for a,b in zip(var, wgts)]), GRB.MAXIMIZE)
# Add constraint: x + 2 y + 3 z <= 4
m.addConstr(b == quicksum([a*b for a,b in zip(var, gas)]))
m.addConstr(b <=2.5, "c0")
for v in var:
    m.addConstr(v<=1)

# Add constraint: x + y >= 1
m.optimize()
for v in m.getVars():
    print(v.varName, v.x)

print('Obj:', m.objVal)

for c in m.getConstrs():    
    print("constr", c.ConstrName, "slack ", c.slack)