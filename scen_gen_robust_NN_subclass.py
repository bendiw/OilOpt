# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 15:08:13 2018

@author: bendi
"""

# -*- coding: utf-8 -*f-

# =============================================================================
# Nomenclature:
#   Thesis  |   Code
#   x       |    mu
#   s       |    rho
#   z       |    lambda
# =============================================================================
"""
Created on Wed Feb 21 10:33:12 2018

@author: bendiw
"""
from gurobipy import *
import numpy as np
import math
import pandas as pd
import tools as t
import os.path
from recourse_model import Recourse_Model

class NN(Recourse_Model):
    
    # =============================================================================
    # get neural nets either by loading existing ones or training new ones
    # =============================================================================
    def getNeuralNets(self, mode, case, phase, net_type="scen"):
        if phase=="gas":
            weights = {scenario : {well : {} for well in self.wellnames} for scenario in range(self.scenarios)}
            biases = {scenario : {well : {} for well in self.wellnames}for scenario in range(self.scenarios)}
            multidims = {scenario : {well : {} for well in self.wellnames} for scenario in range(self.scenarios) }
            layers = {scenario: {well : {} for well in self.wellnames}for scenario in range(self.scenarios)}
            for scenario in range(self.scenarios):
                for well in self.wellnames:
                    weights[scenario][well] = {}
                    biases[scenario][well] = {}
                    multidims[scenario][well] = {}
                    layers[scenario][well] = {} 
                    for separator in self.well_to_sep[well]:
                        if mode==self.LOAD:
                            multidims[scenario][well][separator], weights[scenario][well][separator], biases[scenario][well][separator] = t.load_2(well, phase, separator, case, net_type, scenario)
                            layers[scenario][well][separator] = len(multidims[scenario][well][separator])
                            if net_type=="mean" and multidims[scenario][well][separator][ layers[scenario][well][separator]-1 ] > 1:
                                multidims[scenario][well][separator][ layers[scenario][well][separator]-1 ] -=1
                        else:
                            layers[scenario][well][separator], multidims[scenario][well][separator], weights[scenario][well][separator], biases[scenario][well][separator] = t.train(well, phase, separator, case)
        elif phase == "oil":
            weights = {well : {} for well in self.wellnames}
            biases = {well : {} for well in self.wellnames}
            multidims = {well : {} for well in self.wellnames}
            layers = {well : {} for well in self.wellnames}
            for well in self.wellnames:
                weights[well] = {}
                biases[well] = {}
                multidims[well] = {}
                layers[well] = {} 
                for separator in self.well_to_sep[well]:
                    if mode==self.LOAD:
                        multidims[well][separator], weights[well][separator], biases[well][separator] = t.load_2(well, phase, separator, case, net_type, scen=0)
                        layers[well][separator] = len(multidims[well][separator])
                        if net_type=="mean" and multidims[well][separator][layers[well][separator]-1 ] > 1:
                            multidims[well][separator][ layers[well][separator]-1 ] -=1
                    else:
                        layers[well][separator], multidims[well][separator], weights[well][separator], biases[well][separator] = t.train(well, phase, separator, case)
            
        return layers, multidims, weights, biases
    
    
    # =============================================================================
    # Function to build a model with all wells in the problem.
    # =============================================================================
    def init(self, case=2,
                num_scen = 10, lower=-4, upper=4, phase="gas", sep="HP", save=True,store_init=False, init_name=None, 
                max_changes=15, w_relative_change=None, stability_iter=None, distr="truncnorm", lock_wells=None, scen_const=None, recourse_iter=False, verbose=1):

        Recourse_Model.init(self, case, num_scen, lower, upper, phase, sep, save, store_init, init_name, max_changes, w_relative_change, stability_iter, distr, lock_wells, scen_const, recourse_iter, verbose)        


        self.results_file = "results/robust/res_NN.csv"
        
        #load mean and variance networks
        self.layers_gas, self.multidims_gas, self.weights_gas, self.biases_gas = self.getNeuralNets(self.LOAD, case, net_type="scen", phase="gas")
        self.layers_oil, self.multidims_oil, self.weights_oil, self.biases_oil = self.getNeuralNets(self.LOAD, case, net_type="scen", phase="oil")

            
        # =============================================================================
        # variable creation                    
        # Gas networks scenario dependent, oil networks are not
        # =============================================================================
        self.lambdas_gas = self.m.addVars([(scenario, well,sep, layer, neuron) for scenario in range(self.scenarios) for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_gas[scenario][well][sep]-1) for neuron in range(self.multidims_gas[scenario][well][sep][layer])], vtype = GRB.BINARY, name="lambda_gas")
        self.mus_gas = self.m.addVars([(scenario, well,sep,layer, neuron) for scenario in range(self.scenarios)  for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_gas[scenario][well][sep]-1) for neuron in range(self.multidims_gas[scenario][well][sep][layer])], vtype = GRB.CONTINUOUS, name="mu_gas")
        self.rhos_gas = self.m.addVars([(scenario, well,sep,layer, neuron) for scenario in range(self.scenarios) for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_gas[scenario][well][sep]-1) for neuron in range(self.multidims_gas[scenario][well][sep][layer])], vtype = GRB.CONTINUOUS, name="rho_gas")
        
        self.lambdas_oil = self.m.addVars([(well,sep, layer, neuron) for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_oil[well][sep]-1) for neuron in range(self.multidims_oil[well][sep][layer])], vtype = GRB.BINARY, name="lambda_oil")
        self.mus_oil = self.m.addVars([(well,sep,layer, neuron)  for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_oil[well][sep]-1) for neuron in range(self.multidims_oil[well][sep][layer])], vtype = GRB.CONTINUOUS, name="mu_oil")
        self.rhos_oil = self.m.addVars([(well,sep,layer, neuron)   for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_oil[well][sep]-1) for neuron in range(self.multidims_oil[well][sep][layer])], vtype = GRB.CONTINUOUS, name="rho_oil")

        # =============================================================================
        # NN MILP constraints creation. Mean and variance networks.
        # Variables mu and rho are indexed from 1 and up since the input layer
        # is layer 0 in multidims. 
        # =============================================================================
        # gas
        self.m.addConstrs(self.mus_gas[scenario, well, sep, 1, neuron] - self.rhos_gas[scenario, well, sep, 1, neuron] - quicksum(self.weights_gas[scenario][well][sep][0][dim][neuron]*self.inputs[well, sep, dim] for dim in range(self.multidims_gas[scenario][well][sep][0])) == self.biases_gas[scenario][well][sep][0][neuron] for scenario in range(self.scenarios) for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(self.multidims_gas[scenario][well][sep][1]))
        self.m.addConstrs(self.mus_gas[scenario, well, sep, layer, neuron] - self.rhos_gas[scenario, well, sep, layer, neuron] - quicksum((self.weights_gas[scenario][well][sep][layer-1][dim][neuron]*self.mus_gas[scenario, well, sep, layer-1, dim]) for dim in range(self.multidims_gas[scenario][well][sep][layer-1])) == self.biases_gas[scenario][well][sep][layer-1][neuron] for scenario in range(self.scenarios) for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(2, self.layers_gas[scenario][well][sep]-1) for neuron in range(self.multidims_gas[scenario][well][sep][layer]) )
       
        # oil
        self.m.addConstrs(self.mus_oil[well, sep, 1, neuron] - self.rhos_oil[well, sep, 1, neuron] - quicksum(self.weights_oil[well][sep][0][dim][neuron]*self.inputs[well, sep, dim] for dim in range(self.multidims_oil[well][sep][0])) == self.biases_oil[well][sep][0][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for neuron in range(self.multidims_oil[well][sep][1]) )
        self.m.addConstrs(self.mus_oil[well, sep, layer, neuron] - self.rhos_oil[well, sep, layer, neuron] - quicksum((self.weights_oil[well][sep][layer-1][dim][neuron]*self.mus_oil[well, sep, layer-1, dim]) for dim in range(self.multidims_oil[well][sep][layer-1])) == self.biases_oil[well][sep][layer-1][neuron] for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(2, self.layers_oil[well][sep]-1) for neuron in range(self.multidims_oil[well][sep][layer]) )
        
        # =============================================================================
        # lambda indicator NN constraints        
        # =============================================================================
        #gas
        self.m.addConstrs( (self.lambdas_gas[scenario, well, sep, layer, neuron] == 1) >> (self.mus_gas[scenario, well, sep, layer, neuron] <= 0) for scenario in range(self.scenarios)  for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_gas[scenario][well][sep]-1) for neuron in range(self.multidims_gas[scenario][well][sep][layer]))
        self.m.addConstrs( (self.lambdas_gas[scenario, well, sep, layer, neuron] == 0) >> (self.rhos_gas[scenario, well, sep, layer, neuron] <= 0) for scenario in range(self.scenarios) for phase in self.phasenames for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1,self.layers_gas[scenario][well][sep]-1) for neuron in range(self.multidims_gas[scenario][well][sep][layer]))

        #oil
        self.m.addConstrs( (self.lambdas_oil[well, sep, layer, neuron] == 1) >> (self.mus_oil[well, sep, layer, neuron] <= 0)   for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_oil[well][sep]-1) for neuron in range(self.multidims_oil[well][sep][layer]))
        self.m.addConstrs( (self.lambdas_oil[well, sep, layer, neuron] == 0) >> (self.rhos_oil[well, sep, layer, neuron] <= 0)   for well in self.wellnames for sep in self.well_to_sep[well] for layer in range(1, self.layers_oil[well][sep]-1) for neuron in range(self.multidims_oil[well][sep][layer]))

        #use these to model last layer as linear instead of ReLU
        #gas
        self.m.addConstrs( (self.routes[well, sep] == 1) >> (self.outputs_gas[scenario, well, sep] ==  quicksum(self.weights_gas[scenario][well][sep][self.layers_gas[scenario][well][sep]-2][neuron][0] * self.mus_gas[scenario, well, sep, self.layers_gas[scenario][well][sep]-2, neuron] for neuron in range(self.multidims_gas[scenario][well][sep][self.layers_gas[scenario][well][sep]-2]) ) + self.biases_gas[scenario][well][sep][self.layers_gas[scenario][well][sep]-2][0]) for scenario in range(self.scenarios) for well in self.wellnames for sep in self.well_to_sep[well] )
        self.m.addConstrs( (self.routes[well, sep] == 0) >> (self.outputs_gas[scenario, well, sep] == 0) for scenario in range(self.scenarios) for well in self.wellnames for sep in self.well_to_sep[well])
    
        #oil
        self.m.addConstrs( (self.routes[well, sep] == 1) >> (self.outputs_oil[well, sep] == quicksum(self.weights_oil[well][sep][self.layers_oil[well][sep]-2][neuron][0] * self.mus_oil[well, sep, self.layers_oil[well][sep]-2, neuron] for neuron in range(self.multidims_oil[well][sep][self.layers_oil[well][sep]-2]) ) + self.biases_oil[well][sep][self.layers_oil[well][sep]-2][0]) for well in self.wellnames for sep in self.well_to_sep[well])
        self.m.addConstrs( (self.routes[well, sep] == 0) >> (self.outputs_oil[well, sep] == 0) for well in self.wellnames for sep in self.well_to_sep[well] )
        
        
        return self


    def get_solution(self):        
        df = pd.DataFrame(columns=t.robust_res_columns_SOS2) 
        chokes = [self.inputs[well, "HP", 0].x if self.outputs_gas[0, well, "HP"].x>0 else 0 for well in self.wellnames]
        rowlist=[]
        if(self.case==2 and self.save):
            gas_mean = np.zeros(len(self.wellnames))
#            gas_var=[]
            w = 0
            for well in self.wellnames:
                for scenario in range(self.scenarios):
                    gas_mean[w] += self.outputs_gas[scenario, well, "HP"].x
#                gas_var.append(sum(self.weights_var[well]["gas"]["HP"][self.layers_var[well]["gas"]["HP"]-2][neuron][0] * (self.mus_var[well, "gas", "HP", self.layers_var[well]["gas"]["HP"]-2, neuron].x) for neuron in range(self.multidims_var[well]["gas"]["HP"][self.layers_var[well]["gas"]["HP"]-2])) + self.biases_var[well]["gas"]["HP"][self.layers_var[well]["gas"]["HP"]-2][0])

                w += 1
            gas_mean = (gas_mean/float(self.scenarios)).tolist()
            oil_mean = [self.outputs_oil[well, "HP"].x for well in self.wellnames]
#            oil_var = [outputs_]
            tot_oil = sum(oil_mean)
            tot_gas = sum(gas_mean)
            change = [abs(self.changes[w, "HP", 0].x) for w in self.wellnames]
#            print(df.columns)
            rowlist = [self.scenarios, self.tot_exp_cap, self.well_cap, tot_oil, tot_gas]+chokes+gas_mean+oil_mean+change
#            print(len(rowlist))
            if(self.store_init):
                df.rename(columns={"scenarios": "name"}, inplace=True)
                rowlist[0] = self.init_name
                df.loc[df.shape[0]] = rowlist

                head = not os.path.isfile(self.results_file_init)
                with open(self.results_file_init, 'a') as f:
                    df.to_csv(f, sep=';', index=False, header=head)
            else:
                df.loc[df.shape[0]] = rowlist
                head = not os.path.isfile(self.results_file)
                with open(self.results_file, 'a') as f:
                    df.to_csv(f, sep=';', index=False, header=head)
        elif self.recourse_iter:
            oil_mean = [self.outputs_oil[well, "HP"].x for well in self.wellnames]
            gas_mean = []
#            gas_var = []
            for well in self.wellnames:
                gas_mean.append(sum(self.weights[well]["gas"]["HP"][self.layers[well]["gas"]["HP"]-2][neuron][0] * self.mus[well, "gas", "HP", self.layers[well]["gas"]["HP"]-2, neuron].x for neuron in range(self.multidims[well]["gas"]["HP"][self.layers[well]["gas"]["HP"]-2]) ) + self.biases[well]["gas"]["HP"][self.layers[well]["gas"]["HP"]-2][0] if self.outputs_gas[0, well, "HP"].x>0 else 0)
#                gas_var.append(sum(self.weights_var[well]["gas"]["HP"][self.layers_var[well]["gas"]["HP"]-2][neuron][0] * (mus_var[well, "gas", "HP", self.layers_var[well]["gas"]["HP"]-2, neuron].x) for neuron in range(self.multidims_var[well]["gas"]["HP"][self.layers_var[well]["gas"]["HP"]-2])) + self.biases_var[well]["gas"]["HP"][self.layers_var[well]["gas"]["HP"]-2][0] if self.outputs_gas[0, well, "HP"].x>0 else 0)
            tot_oil = sum(oil_mean)
            tot_gas = sum(gas_mean)+sum([gas_var[w]*self.s_draw.loc[s][self.wellnames[w]] for w in range(len(self.wellnames)) for s in range(self.scenarios)])/self.scenarios
            change = [abs(self.changes[w, "HP", 0].x) for w in self.wellnames]
            rowlist = [self.tot_exp_cap, self.well_cap, tot_oil, tot_gas]+chokes+gas_mean+oil_mean+change
        return df

