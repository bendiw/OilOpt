# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 12:34:20 2018

@author: bendiw
"""

from gurobipy import *
import numpy as np
import tens
import math

# =============================================================================
# get neural nets either by loading existing ones or training new ones
# =============================================================================
def getNeuralNets(mode):
    weights = {well : {} for well in wellnames}
    biases = {well : {} for well in wellnames}
    multidims = {well : {} for well in wellnames}
    for platform in platforms:
        for well in p_dict[platform]:
            for phase in phasenames:
                weights[well][phase] = {}
                biases[well][phase] = {}
                multidims[well][phase] = {}
                for separator in p_sep_names[platform]:
                    if mode==LOAD:
#                        print(well, separator)
                        multidims[well][phase][separator], weights[well][phase][separator], biases[well][phase][separator] = tens.load(well, phase, separator)
                    else:
                        multidims[well][phase][separator], weights[well][phase][separator], biases[well][phase][separator] = tens.train(well, phase, separator)
    return multidims, weights, biases



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

maxouts = ["maxout_1", "maxout_2"]
input_names = ["input_1", "input_2"]


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
w_max_glift = {"A2":{"HP":124200.2899}, "A3":{"HP":99956.56739}, "A5":{"HP":125615.4024}, "A6":{"HP":150090.517}, "A7":{"HP":95499.28792}, "A8":{"HP":94387.68607}, "B1":{"HP":118244.94, "LP":118244.94}, 
               "B2":{"HP":112660.5625, "LP":112660.5625}, "B3":{"HP":138606.6016, "LP":138606.6016},
               "B4":{"HP":90000.0709, "LP":90000.0709}, "B5":{"HP":210086.0959, "LP":210086.0959}, "B6":{"HP":117491.1591, "LP":117491.1591}, "B7":{"HP":113035.4286, "LP":113035.4286}, 
               "C1":{"LP":106860.5264}, "C2":{"LP":132718.54}, "C3" : {"LP":98934.12}, "C4":{"LP":124718.303}}

w_max_lims = [w_max_glift, {well:{sep:100 for sep in well_to_sep[well]} for well in wellnames}]

print(w_max_lims[1])



#dict with allowed relative change, {well: [glift_delta, choke_delta]}
w_relative_change = {well : [1.0, 1.0] for well in wellnames}

#Case relevant numerics
glift_caps = [2600000]
tot_exp_cap = 50000000
sep_cap = {"LP": 20000000, "HP":math.inf}
glift_groups = ["A", "B"]
max_changes = 5

#for w in wellnames:
#    for phase in phasenames:
#        for sep in well_to_sep[w]:
#            tens.hey(w, goal=phase, hp=separator_dict[sep], save=True)

# =============================================================================
# initialize an optimization model
# =============================================================================
m = Model("Ekofisk")

multidims, weights, biases = getNeuralNets(LOAD)

w_route_flow_vars = {}
w_route_bin_vars = {}

input_dict = {(well, sep, dim) : w_max_lims[dim][well][sep]  for well in wellnames for sep in well_to_sep[well] for dim in range(multidims[well]["oil"][sep])}

# =============================================================================
# big-M
# =============================================================================
alpha_M = {well : {phase : {sep : {maxout : [10000000 for n in range(len(biases[well][phase][sep][maxout]))] for maxout in maxouts} for sep in well_to_sep[well]} for phase in phasenames} for well in wellnames}
beta_M = {well : {phase : {sep : 1000000 for sep in well_to_sep[well]} for phase in phasenames} for well in wellnames}             
                    

# =============================================================================
# variable creation                    
# =============================================================================
#inputs = m.addVars([(well,sep, dim) for well in wellnames for sep in well_to_sep[well] for dim in range(multidims[well]["gas"][sep])], vtype = GRB.CONTINUOUS, name="input")
inputs = m.addVars(input_dict.keys(), ub = input_dict, name="input", vtype=GRB.CONTINUOUS)
routes = m.addVars([(well, sep) for well in wellnames for sep in well_to_sep[well]], vtype = GRB.BINARY, name="routing")
betas = m.addVars([(well,phase,sep)  for phase in phasenames for well in wellnames for sep in well_to_sep[well]], vtype = GRB.CONTINUOUS, name="beta") #, lb = -math.inf
alphas = m.addVars([(well,phase,sep, maxout)  for phase in phasenames for well in wellnames for sep in well_to_sep[well] for maxout in maxouts], vtype = GRB.CONTINUOUS, lb=-math.inf, name="alpha")
lambdas = m.addVars([(well,phase,sep, maxout, neuron)  for phase in phasenames for well in wellnames for sep in well_to_sep[well] for maxout in maxouts for neuron in range(len(biases[well][phase][sep][maxout]))], vtype = GRB.BINARY, name="lambda")
mus = m.addVars([(well,phase,sep, maxout, neuron)  for phase in phasenames for well in wellnames for sep in well_to_sep[well] for maxout in maxouts for neuron in range(len(biases[well][phase][sep][maxout]))], vtype = GRB.CONTINUOUS, lb = -math.inf, name="mu")

#print(inputs)
#changes = m.addVars([(well, sep, dim) for well in wellnames for sep in well_to_sep[well] for dim in range(multidims[well]["oil"][sep])], vtype=GRB.BINARY, name="changes")

# =============================================================================
# NN MILP constraints creation
# =============================================================================
#neuron output constraints 7.2
m.addConstrs(mus[well, phase, sep, maxout, neuron] - quicksum(weights[well][phase][sep][maxout][dim][neuron]*inputs[well, sep, dim] for dim in range(multidims[well][phase][sep])) == biases[well][phase][sep][maxout][neuron] for phase in phasenames for well in wellnames for sep in well_to_sep[well] for maxout in maxouts for neuron in range(len(biases[well][phase][sep][maxout])) )

#maxout convexity constraint 7.3
m.addConstrs(quicksum(lambdas[well, phase, sep, maxout, neuron] for neuron in range(len(biases[well][phase][sep][maxout]))) == routes[well, sep] for phase in phasenames for well in wellnames for sep in well_to_sep[well] for maxout in maxouts)

#alpha geq constraint 7.4
m.addConstrs(alphas[well, phase, sep, maxout] >= mus[well, phase, sep, maxout, neuron] for phase in phasenames for well in wellnames for sep in well_to_sep[well] for maxout in maxouts for neuron in range(len(biases[well][phase][sep][maxout])) )

#alpha leq constraint 7.5
#alpha_M[well][phase][sep][maxout][neuron]
m.addConstrs(alphas[well, phase, sep, maxout] + (lambdas[well, phase, sep, maxout, neuron] - 1)*100000000 <= mus[well, phase, sep, maxout, neuron] for phase in phasenames for well in wellnames for sep in well_to_sep[well] for maxout in maxouts for neuron in range(len(biases[well][phase][sep][maxout])) )


#beta value constraint 7.6
m.addConstrs(betas[well, phase, sep] <= alphas[well, phase, sep, maxouts[0]] - alphas[well, phase, sep, maxouts[1]] for phase in phasenames for well in wellnames for sep in well_to_sep[well])

#beta big-M constraint 7.7
m.addConstrs(betas[well, phase, sep] - routes[well, sep]*beta_M[well][phase][sep] <= 0 for phase in phasenames for well in wellnames for sep in well_to_sep[well])


# =============================================================================
# change tracking and total changes
# =============================================================================
#m.addConstrs(w_initial_vars[well][dim] - inputs[well, sep, dim] <= changes[well, sep, dim]*w_initial_vars[well][dim]*w_relative_change[well][dim] for well in wellnames for sep in well_to_sep[well] for dim in range(multidims[well]["oil"][sep]))
#m.addConstrs(inputs[well, sep, dim] - w_initial_vars[well][dim] <= changes[well, sep, dim]*w_initial_vars[well][dim]*w_relative_change[well][dim]+(1-w_initial_prod[well])*w_max_lims[dim][well]*changes[well, sep, dim] for well in wellnames for sep in well_to_sep[well] for dim in range(multidims[well]["oil"][sep]))
#m.addConstr(quicksum(changes[well, sep, dim] for well in wellnames for sep in well_to_sep[well] for dim in range(multidims[well]["oil"][sep])) <= max_changes)

# =============================================================================
# separator gas constraints
# =============================================================================
m.addConstr(quicksum(betas[well, "gas", "LP"] for w in sep_p_route["LP"] for well in p_dict[w] ) - quicksum(inputs[c_well, "LP", 0] for c_well in p_dict["C"]) <= sep_cap["LP"])
m.addConstr(quicksum(betas[well, "gas", "HP"] for w in sep_p_route["HP"] for well in p_dict[w]) <= sep_cap["HP"])


# =============================================================================
# gas lift constraints
# =============================================================================
m.addConstr(quicksum(inputs[well, sep, 0] for pform in glift_groups for well in p_dict[pform] for sep in well_to_sep[well]) <= glift_caps[0])


# =============================================================================
# total gas export
# =============================================================================
m.addConstr(quicksum(betas[well, "gas", sep] for well in wellnames for sep in well_to_sep[well]) - quicksum(inputs[c_well, "LP", 0] for c_well in p_dict["C"]) <= tot_exp_cap)


# =============================================================================
# routing
# =============================================================================
m.addConstrs(quicksum(routes[well, sep] for sep in well_to_sep[well]) <= 1 for well in wellnames)


# =============================================================================
# objective
# =============================================================================
m.setObjective(quicksum(betas[well, "oil", sep] for well in wellnames for sep in well_to_sep[well]), GRB.MAXIMIZE)
m.setParam(GRB.Param.Heuristics, 0)
m.setParam(GRB.Param.Presolve, 0)
m.optimize()

        
for v in m.getVars()[0:150]:
    print(v)

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