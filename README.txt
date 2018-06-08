This file contains information of the .py files used in the development of this thesis,as well as
descriptions of their most functions.

caseloader.py
	-Loads well data from .csv file, and returns dictionaries of normalized data.
	
gridsearch.py
	-Uses the NeuralRegressor class to search for optimal parameters in a Gurobi optimization model.
	
heteroscedastic_customloss.py
	-Receives data from caseloader.py and trains neural networks using Keras. Estimates mean and
	prediction variance with epistemic and heteroscedastic aleatoric uncertainty, using a specified loss
	function and structure.
	
MOP.py
	-Receives samples from trained neural networks and performs the e-constrained method in order
	to maximize oil output and minimize the probability of infeasibility.
	
tools.py
	-Holds case specific data and performs general "service"-tasks for the other classes.

recourse_algorithm.py
	-The RA from the thesis is implemented in this file. Performs an optimization run, using
	recourse_models.py, chooses a change and implements it. Then runs new optimization runs.

recourse_models.py
	-This is the optimization models. Solves problems with SOS2 or neural network representation
	of the scenarios, using Gurobi.

NN_scenario.py
	-Generates Factor and Markov Weighted scenarios with or without known operating points.


