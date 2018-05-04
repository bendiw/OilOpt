# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:23:48 2018

@author: bendi
"""

from gurobipy import *
import numpy as np
import math
import pandas as pd
import tools as t
from keras import backend as K
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras import optimizers, regularizers, initializers
import os.path

class Evaluator:
    
    # =============================================================================
    # get NN data from file    
    # =============================================================================
    def getNeuralNetsData(self, case, net_type="mean"):
        weights = {well : {} for well in self.wellnames}
        biases = {well : {} for well in self.wellnames}
        multidims = {well : {} for well in self.wellnames}
        layers = {well : {} for well in self.wellnames}
        for well in self.wellnames:
            for phase in self.phasenames:
                weights[well][phase] = {}
                biases[well][phase] = {}
                multidims[well][phase] = {}
                layers[well][phase] = {} 
                for separator in self.well_to_sep[well]:
                    multidims[well][phase][separator], w, biases[well][phase][separator] = t.load_2(well, phase, separator, case, net_type)
                    weights[well][phase][separator] = []
#                    print("\n\n",([[x[0]] for x in w[-1]]))
#                    print(w[-1])
#                    print("\n\n",(biases[well][phase][separator][-1]))
#                    print("\n\n",(w[1]))
#                    print("\n\n",(biases[well][phase][separator][1]))
                    for layer in range(len(w)-1):
                        weights[well][phase][separator].append([np.array(w[layer]), np.array(biases[well][phase][separator][layer])])
                    
                    #remove weights, bias for second output neuron in case it exists
                    if(multidims[well][phase][separator][-1] ==2):
                        weights[well][phase][separator].append([np.array([[x[0]] for x in w[-1]]), np.array([biases[well][phase][separator][-1][0]])])
                    else:
                        weights[well][phase][separator].append([np.array(w[-1]), np.array(biases[well][phase][separator][-1])])


                    if net_type=="mean" and multidims[well][phase][separator][-1] > 1:
                        multidims[well][phase][separator][-1] -=1
        return multidims, weights
    
    
    # =============================================================================
    # builds single NN. Assume for now ReLU in hidden layers and linear outputs    
    # =============================================================================
    def build_model(self, m_multidims, m_weights):
        m= Sequential()
        m.add(Dense(m_multidims[1], input_shape=(m_multidims[0],), weights=m_weights[0]))
        m.add(Activation("relu"))
        for i in range(2, len(m_multidims)-1):
            m.add(Dense(m_multidims[i], weights=m_weights[i-1])) 
            m.add(Activation("relu"))
    
        m.add(Dense(m_multidims[-1], weights=m_weights[-1]))
        m.add(Activation("linear"))
        m.compile(optimizer=optimizers.Adam(lr=0), loss="mse")
        return m
    
    # =============================================================================
    # build all NNs    
    # =============================================================================
    def buildNeuralNets(self, multidims, weights):
        nets = {w:{} for w in self.wellnames}
        for well in self.wellnames:
            for phase in self.phasenames:
                nets[well][phase] = {}
                for sep in self.well_to_sep[well]:
#                    print(well, phase, sep)
#                    print(multidims.keys())
#                    print("\n\n", weights.keys())
#                    print(nets.keys())
                    nets[well][phase][sep] = self.build_model(multidims[well][phase][sep], weights[well][phase][sep])
        return nets
    
    
    # =============================================================================
    # evaluate solution in a single scenario, no recourse allowed
    # =============================================================================
    def calc_results(self, s):
        gas_mean = []
        oil_mean = []
        gas_var = []
        oil_var = []
        tot_gas = 0
        inf_indiv = 0
        inf_tot = 0
        for w in self.wellnames:
            well_mean_gas = self.nets_mean[w]["gas"]["HP"].predict(self.solution[w])[0][0]
            well_var_gas = self.nets_var[w]["gas"]["HP"].predict(self.solution[w])[0][0]*s[w]
#            print(well_mean_gas)
#            print(well_var_gas)
            well_tot_gas = well_mean_gas + well_var_gas
#            print(well_tot_gas)
            if(well_tot_gas > self.indiv_cap):
                inf_indiv = 1
            gas_mean.append(well_mean_gas)
            oil_mean.append(self.nets_mean[w]["oil"]["HP"].predict(self.solution[w])[0][0])
            gas_var.append(well_var_gas)
            oil_var.append(self.nets_var[w]["oil"]["HP"].predict(self.solution[w])[0][0]*s[w])
            tot_gas += well_tot_gas
        
        tot_oil = sum(oil_mean)
        if(tot_gas>self.tot_cap):
            inf_tot = 1
#        print(tot_gas.shape)
#        print(tot_oil.shape)
#        return 
        r = [inf_tot] + [inf_indiv] + [tot_oil] +[tot_gas]+ gas_mean + oil_mean + oil_var  + gas_var
        return r
#robust_eval_columns = ["tot_oil", "tot_gas"]+[w+"_gas_mean" for w in wellnames_2]+[w+"_oil_mean" for w in wellnames_2]+[w+"_oil_var" for w in wellnames_2]+[w+"_gas_var" for w in wellnames_2]


    # =============================================================================
    # Calculate results for single scenario where recourse is allowed. That is, 
    # if a constraint is reached, we stop increasing choke values to avoid
    # infeasibility.
    # =============================================================================
    def calc_results_recourse(self, s):
        pass
    
    # =============================================================================
    # main function    
    # =============================================================================
    def evaluate(self, problem, case=2, sol_scen=100, eval_scen=10000):
        self.results_file = "results/robust/res_eval"
        if case==2:
            self.wellnames = t.wellnames_2
            self.well_to_sep = t.well_to_sep_2
            self.phasenames = t.phasenames
        else:
            pass
        print("loading scenarios...")
        self.scenarios = t.get_scenario(case=case, num_scen=eval_scen)
        self.solution, self.indiv_cap, self.tot_cap = t.get_robust_solution(sol_scen)
        multidims, weights = self.getNeuralNetsData(case=case)
        multidims_var, weights_var = self.getNeuralNetsData(case=case, net_type="var")
        print("building models...")
        self.nets_mean = self.buildNeuralNets(multidims, weights)
        print("mean networks done.")
        self.nets_var = self.buildNeuralNets(multidims_var, weights_var)
        print("var networks done.")
        df = pd.DataFrame(columns=t.robust_eval_columns)
        print("evaluating scenarios...")
        
        self.GOR_means()
        
#        for s in range(eval_scen):
#            r = self.calc_results(self.scenarios.loc[s])
#            df.loc[s] = r
##            print("\n\n\n",df.loc[s])
#        #TODO: add case specific info to filename
#        with open(self.results_file+problem+".csv", "w") as f:
#            df.to_csv(f, sep=';', index=False)
            
    # =============================================================================
    # helper function to determine GOR for each well. use this to select order of
    # wells to switch on in recourse evaluation    
    # =============================================================================
    def GOR_means(self):
        w_min_choke, w_max_choke = t.get_limits("choke", self.wellnames, self.well_to_sep, case)
        ranges = {w : np.arange(w_min_choke[w], w_max_choke[well], 100) for w in self.wellnames}
        oil = {w:[] for w in self.wellnames}
        gas = {w:[] for w in self.wellnames}
        ratios = {w:[] for w in self.wellnames}
        for w in self.wellnames:
            oil[w] = self.nets_mean[w]["oil"]["HP"].predict(ranges[w])
            gas[w] = self.nets_mean[w]["gas"]["HP"].predict(ranges[w])
            ratios[w] = np.mean(oil[w] / gas[w])
        print(ratios)
            