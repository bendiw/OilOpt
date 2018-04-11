# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 13:50:24 2018

@author: bendi
"""

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf



@tf.RegisterGradient("CustomClipGrad")
def _clip_grad(unused_op, grad):
  return tf.clip_by_value(grad, -0.1, 0.1)

input = tf.Variable([3.0], dtype=tf.float32)

g = tf.get_default_graph()
with g.gradient_override_map({"Identity": "CustomClipGrad"}):
  output_clip = tf.identity(input, name="Identity")
grad_clip = tf.gradients(output_clip, input)

# output without gradient clipping in the backwards pass for comparison:
output = tf.identity(input)
grad = tf.gradients(output, input)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print("with clipping:", sess.run(grad_clip)[0])
  print("without clipping:", sess.run(grad)[0])