# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 13:50:24 2018

@author: bendi
"""

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf


# =============================================================================
# calculates new weights for the HeteroScedastic regression layer
# =============================================================================
def calcWeights(outputs, y, old_weights, old_biases):
    pass

def get_trainable_params(model):
    params = []
    for layer in model.layers:
        params += keras.engine.training.collect_trainable_weights(layer)
    return params
    

model = Sequential()
##add layers

network_params = get_trainable_params(model)
param_grad = tf.gradients(cost, network_params)
param_grad = sess.run(param_grad_sym, \
                       feed_dict={x: input_vals, model_params: model.get_weights()})
new_weights = custom_optimization_routine(param_grad, model.get_weights(), other_args)
model.set_weight(new_weights)