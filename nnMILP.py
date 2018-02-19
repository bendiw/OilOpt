# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 12:34:20 2018

@author: bendiw
"""

from gurobipy import *
import numpy as np
import tens
import math


phasenames = ["oil", "gas"]
sepnames = ["LP", "HP"]
invars = ["gas lift", "choke"]
OIL = 0
GAS = 1
LOAD = 0
TRAIN = 1
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

#dict with binary var describing whether or not wells are producing in initial setting
w_initial_prod = {well : 0 for well in wellnames}


#dict with initial values for choke, gas lift per well, {well: [gas lift, choke]}
w_initial_vars = {well : [0,0] for well in wellnames}

#dict with maximum gaslift in each well
w_max_glift = {"A2":124200.2899, "A3":99956.56739, "A5":125615.4024, "A6":150090.517, "A7":95499.28792, "A8":94387.68607, "B1":118244.94, "B2":112660.5625, "B3":138606.6016,
               "B4":197509.0709, "B5":210086.0959, "B6":117491.1591, "B7":125035.4286, "C1":106860.5264, "C2":132718.54, "C3" : 98934.12, "C4":124718.303}

w_max_lims = [w_max_glift, {well:1 for well in wellnames}]

#dict with allowed relative change, {well: [glift_delta, choke_delta]}
w_relative_change = {well : [1.0, 1.0] for well in wellnames}

#Case relevant numerics
glift_caps = [260000]
tot_exp_cap = 500000
sep_cap = {"LP": 200000, "HP":math.inf}
glift_groups = [["A", "B"]]
max_changes = 5

# =============================================================================
# initialize an optimization model
# =============================================================================
m = Model("Ekofisk")

#multidims, weights, biases = getNeuralNets(mode)

w_route_flow_vars = {}
w_route_bin_vars = {}


# =============================================================================
# big-M
# =============================================================================
alpha_M = {}
beta_M = {}             
                    

# =============================================================================
# variable creation                    
# =============================================================================
betas = m.addVars([(well,phase,sep)  for phase in range(len(phasenames)) for well in wellnames for sep in well_to_sep[well]], vtype = GRB.CONTINUOUS, name="beta")
alphas = m.addVars([(well,phase,sep, maxout)  for phase in range(len(phasenames)) for well in wellnames for sep in well_to_sep[well] for maxout in range(2)], vtype = GRB.CONTINUOUS, name="alpha")
lambdas = m.addVars([(well,phase,sep, maxout, neuron)  for phase in phasenames for well in wellnames for sep in well_to_sep[well] for maxout in range(2) for neuron in biases[well][phase][sep][maxout]], vtype = GRB.BINARY, name="lambda")
mus = m.addVars([(well,phase,sep, maxout, neuron)  for phase in phasenames for well in wellnames for sep in well_to_sep[well] for maxout in range(2) for neuron in biases[well][phase][sep][maxout]], vtype = GRB.CONTINUOUS, name="mu")
inputs = m.addVars([(well,sep, dim) for well in wellnames for sep in well_to_sep[well] for dim in range(multidims[well][phase][sep])], vtype = GRB.CONTINUOUS, name="input")

routes = m.addVars([(well, sep) for well in wellnames for sep in well_to_sep[well]], vtype = GRB.BINARY, name="routing")
changes = m.addVars([(well, sep, dim)] for well in wellnames for sep in well_to_sep[well] for dim in range(multidims[well][0][sep]))


# =============================================================================
# NN MILP constraints creation
# =============================================================================
#neuron output constraints 7.2
m.addConstrs(mus[well, phase, sep, maxout, neuron] - quicksum(weights[well][phase][separator][maxout][dim][neuron]*inputs[well, separator, dim] for dim in range(multidims[well][phase][sep])) == biases[well][phase][separator][maxout][dim][neuron] for phase in phasenames for well in wellnames for sep in well_to_sep[well] for maxout in range(2) for neuron in biases[well][phase][sep][maxout] )

#maxout convexity constraint 7.3
m.addConstrs(quicksum(lambdas[well, phase, sep, maxout, neuron] for neuron in biases[well][phase][sep][maxout]) == 1 for phase in phasenames for well in wellnames for sep in well_to_sep[well] for maxout in range(2))

#alpha geq constraint 7.4
m.addConstr(alphas[well, phase, sep, maxout] >= mus[well, phase, sep, maxout, neuron] for phase in phasenames for well in wellnames for sep in well_to_sep[well] for maxout in range(2) for neuron in biases[well][phase][sep][maxout] )

#alpha leq constraint 7.5
m.addConstrs(alphas[well, phase, sep, maxout] + (lambdas[well, phase, sep, maxout, neuron] - 1)*alpha_M[SOMETHING] <= mus[well, phase, sep, maxout, neuron] for phase in phasenames for well in wellnames for sep in well_to_sep[well] for maxout in range(2) for neuron in biases[well][phase][sep][maxout] )

#beta value constraint 7.6
m.addConstrs(betas[well, phase, sep] == alphas[well, phase, sep, 0] - alphas[well, phase, sep, 0] for phase in phasenames for well in wellnames for sep in well_to_sep[well])

#beta big-M constraint 7.7
m.addConstrs(betas[well, phase, sep] - routes[well, sep]*beta_M[SOMETHING] <= 0 for phase in phasenames for well in wellnames for sep in well_to_sep[well])


# =============================================================================
# change tracking
# =============================================================================
m.addConstrs(w_initial_vars[well][dim] - inputs[well, sep, dim] <= changes[well, sep, dim]*w_initial_vars[well][dim]*w_relative_change[well][dim] for well in wellnames for sep in well_to_sep[well] for dim in range(multidims[well][0][sep]))
m.addConstrs(inputs[well, sep, dim] - w_initial_vars[well][dim] <= changes[well, sep, dim]*w_initial_vars[well][dim]*w_relative_change[well][dim]+(1-w_initial_prod[well])*w_max_lims[dim][well]*changes[well, sep, dim] for well in wellnames for sep in well_to_sep[well] for dim in range(multidims[well][0][sep]))

# =============================================================================
# separator gas constraints
# =============================================================================
m.addConstr(quicksum(betas[well, GAS, "LP"] for w in sep_p_route["LP"] for well in p_dict[w] ) - quicksum(inputs[c_well, "LP", 0] for c_well in p_dict["C"]) <= sep_cap["LP"])
m.addConstr(quicksum(betas[well, GAS, "HP"] for w in sep_p_route["HP"] for well in p_dict[w]) <= sep_cap["HP"])


# =============================================================================
# gas lift constraints
# =============================================================================
m.addConstrs()
for i in range(len(glift_groups)):
    m.addConstr(quicksum([quicksum([quicksum([a*b[0] for a,b in w_breakpoints[OIL][n][z].items()]) for n in p_dict["A"]]) for z in p_sep_names["A"]])
    + quicksum([quicksum([quicksum([a*b[0] for a,b in w_breakpoints[OIL][m][z].items()]) for m in p_dict["B"]]) for z in p_sep_names["B"]])<= glift_caps[i], "glift_a_b")


# =============================================================================
# routing
# =============================================================================
m.addConstrs(quicksum(routes[well, sep] for sep in well_to_sep[well]) <= 1 for well in wellnames)



for well in wellnames:
    if(any(multidims[OIL][well].values())):
        changevar_g = m.addVar(vtype = GRB.BINARY, name=well+"_choke_change_binary")
        w_change_vars[well].append(changevar_g)
        for separator in well_to_sep[well]:
            if(multidims[OIL][well][separator]):
                pass
#                m.addConstr(w_initial_vars[well][1] - quicksum([a*b[1] for a,b in w_breakpoints[OIL][well][separator].items()]) <=  changevar_g*w_initial_vars[well][1]*w_relative_change[well][1])
#                m.addConstr(quicksum([a*b[1] for a,b in w_breakpoints[OIL][well][separator].items()]) - w_initial_vars[well][1] <=  changevar_g*w_initial_vars[well][1]*w_relative_change[well][1]+ (1-w_initial_prod[well])*100*changevar_g)
        
        
        
        
# =============================================================================
# get neural nets either by loading existing ones or training new ones
# =============================================================================
def getNeuralNets(mode):
    weights = {well : {} for well in wellnames}
    biases = {well : {} for well in wellnames}
    multidims = {well : {} for well in wellnames}
    for platform in platforms:
        for well in p_dict[platform]:
            for phase in range(len(phasenames)):
                for separator in p_sep_names[platform]:
                    weights[well][phase] = {}
                    biases[well][phase] = {}
                    multidims[well][phase] = {}
                    if mode==LOAD:
                        multidims[well][phase][separator], weights[well][phase][separator], biases[well][phase][separator] = tens.load(well, phase, separator)
                    else:
                        multidims[well][phase][separator], weights[well][phase][separator], biases[well][phase][separator] = tens.train(well, phase, separator)
    return multidims, weights, biases

# =============================================================================
# old code
# =============================================================================
#routing variables
#for pform in platforms:
#        for well in p_dict[pform]:
#            well_sep_bin = {}#[] was list
#            routes = []
#            for sep in p_sep_names[pform]:
#                routevar = m.addVar(vtype = GRB.BINARY, name=well+"_bin_route_"+sep+"_sep")
#                well_sep_bin[sep] = routevar
#                routes.append(routevar)
#            m.addConstr(quicksum(routes)<=1, well+"_"+"routing_decision")
#            w_route_bin_vars[well] = well_sep_bin
#
#
## dict to store beta output variables 
#betas = {well : {} for well in wellnames}
#
#for platform in platforms:
#        for well in p_dict[platform]:
#            betas[well][phase] = {}
#            for phase in range(len(phasenames)):
#                for separator in p_sep_names[platform]:
#                    alphas = []
#                    for input_unit in ....:
#                        alphas.append(m.addVar(vtype = GRB.CONTINUOUS, name = well+"_"+phasenames[phase]+"_"+separator+"_alpha))
#
#                        for neuron in biases[well][phase][separator][input_unit]:
#                            alphas.append(m.addVar(vtype = GRB.BINARY, name = well+"_"+phasenames[phase]+"_"+separator+"_neuron_binary"))
#                            
#                        
#                    beta = m.addVar(vtype = GRB.BINARY, name=well+"_"+phasenames[phase]+"_ptope_"+separator+"_"+str(p))
#                    betas[well][phase][separator] = beta
#                    m.addConstr(beta = quicksum())