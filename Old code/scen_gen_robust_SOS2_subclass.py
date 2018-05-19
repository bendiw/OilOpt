# -*- coding: utf-8 -*-
"""
Created on Wed May 16 09:58:02 2018

@author: bendiw
"""

# -*- coding: utf-8 -*f-

# =============================================================================
# Nomenclature:
#   Thesis  |   Code
#   x       |    mu
#   s       |    rho
#   z       |    lambda
# =============================================================================
from gurobipy import *
import numpy as np
import math
import pandas as pd
import tools as t
import os.path
from recourse_model import Recourse_Model

class SOS2(Recourse_Model):
    # =============================================================================
    # Function to build model with all wells in the problem.
    # =============================================================================
    def init(self, case=2,
                num_scen = 200, lower=-4, upper=4, phase="gas", sep="HP", save=True,store_init=False, init_name=None, 
                max_changes=15, w_relative_change=None, stability_iter=None, distr="truncnorm", lock_wells=None, scen_const=None, recourse_iter=False, verbose=1, points=10):
        
        Recourse_Model.init(self, case, num_scen, lower, upper, phase, sep, save, store_init, init_name, max_changes, w_relative_change, stability_iter, distr, lock_wells, scen_const, recourse_iter, verbose)        
        self.results_file = "results/robust/res_SOS2.csv"
        #load SOS2 breakpoints
        self.oil_vals = t.get_sos2_scenarios("oil", self.scenarios)
        self.gas_vals = t.get_sos2_scenarios("gas", self.scenarios)
        if(self.scenarios=="eev"):
            self.scenarios=1
        self.choke_vals = [i*100/(len(self.oil_vals["W1"])-1) for i in range(len(self.oil_vals["W1"]))]
#        print(self.oil_vals)
        #workaround to multidims
        self.multidims= {}
        
        # =============================================================================
        # variable creation                    
        # =============================================================================
        #continuous breakpoint variables
        self.zetas = self.m.addVars([(brk, well, sep) for well in self.wellnames for sep in self.well_to_sep[well] for brk in range(len(self.choke_vals))], vtype = GRB.CONTINUOUS, name="zetas")

        # =============================================================================
        # SOS2
        # =============================================================================
        #oil
        self.oil_brks_constr = self.m.addConstrs( (self.routes[well, sep] == 1) >> (self.outputs_oil[well, sep] == quicksum( self.zetas[brk, well, sep]*self.oil_vals[well][brk] for brk in range(len(self.choke_vals)))) for well in self.wellnames for sep in self.well_to_sep[well])
        self.m.addConstrs( (self.routes[well, sep] == 0) >> (self.outputs_oil[well, sep] == 0) for well in self.wellnames for sep in self.well_to_sep[well] )
        #gas
        self.gas_brks_constr = self.m.addConstrs( (self.routes[well, sep] == 1) >> (self.outputs_gas[scenario, well, sep] == quicksum(self.zetas[brk, well, sep]*self.gas_vals[scenario][well][brk] for brk in range(len(self.choke_vals)))) for well in self.wellnames for sep in self.well_to_sep[well] for scenario in range(self.scenarios) )
        self.m.addConstrs( (self.routes[well, sep] == 0) >> (self.outputs_gas[scenario, well, sep] == 0) for scenario in range(self.scenarios) for well in self.wellnames for sep in self.well_to_sep[well] ) 
        # =============================================================================
        # input limit constraints
        # =============================================================================
        self.m.addConstrs( self.inputs[well, sep, 0] == quicksum(self.zetas[brk, well, sep]*self.choke_vals[brk] for brk in range(len(self.choke_vals))) for well in self.wellnames for sep in self.well_to_sep[well])
#        self.m.addConstrs( (self.routes[well, sep] == 0) >> (self.inputs[well, sep, 0] == 0) for well in self.wellnames for sep in self.well_to_sep[well])

        # =============================================================================
        # SOS2 constraints 
        # =============================================================================
        #no way to do in one-liner
        for well in self.wellnames:
            for sep in self.well_to_sep[well]:
                self.m.addSOS(2, [self.zetas[brk, well, sep] for brk in range(len(self.choke_vals))])
        #tighten sos constraints
        self.m.addConstrs( self.routes[well, sep] == quicksum( self.zetas[brk, well, sep] for brk in range(len(self.choke_vals))) for well in self.wellnames for sep in self.well_to_sep[well] for scenario in range(self.scenarios) )
        return self


