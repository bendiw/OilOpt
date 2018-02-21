# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:33:12 2018

@author: bendiw
"""

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
    w_max_glifts = {"A2":{"HP":124200.2899}, "A3":{"HP":99956.56739}, "A5":{"HP":125615.4024}, "A6":{"HP":150090.517}, "A7":{"HP":95499.28792}, "A8":{"HP":94387.68607}, "B1":{"HP":118244.94, "LP":118244.94}, 
               "B2":{"HP":112660.5625, "LP":112660.5625}, "B3":{"HP":238606.6016, "LP":138606.6016},
               "B4":{"HP":90000.0709, "LP":90000.0709}, "B5":{"HP":210086.0959, "LP":210086.0959}, "B6":{"HP":117491.1591, "LP":117491.1591}, "B7":{"HP":113035.4286, "LP":113035.4286}, 
               "C1":{"LP":106860.5264}, "C2":{"LP":132718.54}, "C3" : {"LP":98934.12}, "C4":{"LP":124718.303}}

    # =============================================================================
    # get neural nets either by loading existing ones or training new ones
    # =============================================================================
    def getNeuralNet(self, mode, well, sep):
        weights = {well : {} for well in self.wellnames}
        biases = {well : {} for well in self.wellnames}
        multidims = {well : {} for well in self.wellnames}
        for platform in self.platforms:
            for well in self.p_dict[platform]:
                for phase in self.phasenames:
                    weights[well][phase] = {}
                    biases[well][phase] = {}
                    multidims[well][phase] = {}
                    for separator in self.p_sep_names[platform]:
                        if mode==self.LOAD:
    #                        print(well, separator)
                            multidims[well][phase][separator], weights[well][phase][separator], biases[well][phase][separator] = tens.load(well, phase, separator)
                        else:
                            multidims[well][phase][separator], weights[well][phase][separator], biases[well][phase][separator] = tens.train(well, phase, separator)
        return multidims, weights, biases
    
    
    def run(self, well, sep):
        # =============================================================================
        # initialize an optimization model
        # =============================================================================
        self.m = Model("Ekofisk")
        self.multidims, self.weights, self.biases = self.getNeuralNet(self.LOAD, well, sep)
        
        
        
        w_max_lims = [{self.wellnames[0]:self.w_max_glifts[self.wellnames[0]]}, {well:{sep:100 for sep in self.well_to_sep[well]} for well in self.wellnames}]
        input_dict = {(well, sep, dim) : w_max_lims[dim][well][sep]  for well in self.wellnames for sep in self.well_to_sep[well] for dim in range(self.multidims[well][self.phasenames[0]][sep])}

        # =============================================================================
        # big-M
        # =============================================================================
        alpha_M = {well : {phase : {sep : {maxout : [10000000 for n in range(len(self.biases[well][phase][sep]))]} for sep in self.well_to_sep[well]} for phase in self.phasenames} for well in self.wellnames}
        beta_M = {well : {phase : {sep : 1000000 for sep in self.well_to_sep[well]} for phase in self.phasenames} for well in self.wellnames}             
        
        
        # =============================================================================
        # variable creation                    
        # =============================================================================
        #inputs = m.addVars([(well,sep, dim) for well in wellnames for sep in well_to_sep[well] for dim in range(multidims[well]["gas"][sep])], vtype = GRB.CONTINUOUS, name="input")
        inputs = self.m.addVars(input_dict.keys(), ub = input_dict, name="input", vtype=GRB.CONTINUOUS)
#        routes = self.m.addVars([(well, sep) for well in self.wellnames for sep in self.well_to_sep[well]], vtype = GRB.BINARY, name="routing")
        lambdas = self.m.addVars([(well,phase,sep, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases[well][phase][sep]))], vtype = GRB.BINARY, name="lambda")
        mus = self.m.addVars([(well,phase,sep, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases[well][phase][sep][maxout]))], vtype = GRB.CONTINUOUS, name="mu")
        rhos = self.m.addVars([(well,phase,sep, neuron)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases[well][phase][sep]))], vtype = GRB.CONTINUOUS, name="rho")

        # =============================================================================
        # NN MILP constraints creation
        # =============================================================================
        constr = self.m.addConstrs(mus[well, phase, sep, neuron] - rhos[well, phase, sep, neuron] - quicksum((self.weights[well][phase][sep][dim][neuron]*inputs[well, sep, dim]) for dim in range(self.multidims[well][phase][sep])) == self.biases[well][phase][sep][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases[well][phase][sep])) )
        print(self.weights)

        #indicator constraints
        self.m.addConstrs( (lambdas[well, phase, sep, neuron] == 1) >> (mus[well, phase, sep, neuron] <= 0)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases[well][phase][sep])))
        self.m.addConstrs( (lambdas[well, phase, sep, neuron] == 0) >> (rhos[well, phase, sep, neuron] <= 0)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(len(self.biases[well][phase][sep])))

    
    
        # =============================================================================
        # objective
        # =============================================================================
        self.m.setParam(GRB.Param.NumericFocus, 3)
        self.m.setObjective(quicksum(mus[well, "oil", sep] for well in self.wellnames for sep in self.well_to_sep[well]), GRB.MAXIMIZE)
        self.m.setParam(GRB.Param.Heuristics, 0)
        self.m.setParam(GRB.Param.Presolve, 0)
        self.m.optimize()
    
    
        for v in self.m.getVars()[0:150]:
            print(v)
    
    
    def nn(self, well, sep):
        self.wellnames = [well]
        self.well_to_sep[well]= [sep]
        self.platforms= [well[0]]
        self.p_dict[well[0]] = [well]
        self.p_sep_names[self.platforms[0]] = [sep]
        self.phasenames = ["oil", "gas"]
        self.run(well, sep)