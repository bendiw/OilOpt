# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from matplotlib import pyplot
import caseloader as cl
import tools
import plotter
import tens
from sklearn.preprocessing import RobustScaler

# =============================================================================
# This is the file handling the Tensorflow module, regularization, cross_validation,
# and a good deal of printing methods, designed for a specific purpose.
# =============================================================================

# =============================================================================
# add_layer creates a new Tensorflow layer, with biases and weights uniformly distributed
# between -1 and 1. It performs tf.matmul as default, and any additional activation function
# when it is provided by the caller. input_size and output_size is the number of neurons
#in the previos layer and in the current layer, respectively
# =============================================================================
def add_layer(inputs, input_size, output_size, bias = True, activation_function = None, relu_unit="0"):
    weightname = "weight-"+relu_unit
    biasname = "bias-"+relu_unit
    if (bias == True):
        W = tf.Variable(np.random.uniform(-0.05, 0.05, size = (input_size, output_size)), name=weightname)
        b = tf.Variable(np.random.uniform(0, 0, size =output_size), name=biasname)
        output = tf.matmul(inputs, W) + b
    else:
        W = tf.Variable(np.ones((input_size, output_size)), name=weightname)
        output = tf.matmul(inputs, W)
        b=None
    if activation_function is not None:
        output = activation_function(output)
    print("New layer with activation function", activation_function)
    return output, W, b

def leaky_relu(x):
    return tf.nn.relu(x) - 0.001*tf.nn.relu(-x)

# =============================================================================
#  Initiates a new TF network if an old one is not provided. If an old one is provided, the
# weights and biases are reinitialized. Sends prev back and forth between methods. Prev has
#    all the necessary TF session information
# =============================================================================
def hey(datafile, test_error = False, goal='oil', grid_size = 8, plot = False,learning_rate=0.01, factor = 1.5, cross_validation = None,
        epochs = 100000, beta = 0.01, val_interval = None, train_frac = 1.0, val_frac = 0.0, n_hidden = 40,
        k_prob = 1.0, normalize = False, intervals = 20, nan_ratio = 0.3, hp=1, prev = None, save = False):
    if (prev == None):
        prev = [0]*2
        x = tf.placeholder(tf.float64, [None,1])
        y_ = tf.placeholder(tf.float64, [None,1])
        keep_prob = tf.placeholder(tf.float64)
        L1, W1, b1 = add_layer(x, 1, n_hidden, activation_function = leaky_relu, relu_unit="1-1")
        regularizers = tf.nn.l2_loss(W1)
        L12, W12, b12 = add_layer(L1, n_hidden, 1, activation_function = None, bias=True, relu_unit="1-2")
        out1 = tf.nn.dropout(L12, keep_prob)
        loss = tf.reduce_mean(tf.square(y_ - out1) + beta * regularizers)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess = tf.InteractiveSession()
        prev[0] = [sess, train_step, x, y_, keep_prob, out1, loss]
        print("Initializing Tensorflow modules. This may take a while...")
        
        x2 = tf.placeholder(tf.float64, [None,2])
        y_2 = tf.placeholder(tf.float64, [None,1])
        keep_prob2 = tf.placeholder(tf.float64)
        L2, W2, b2 = add_layer(x2, 2, n_hidden, activation_function = tf.nn.relu, relu_unit="2-1")
        regularizers2 = tf.nn.l2_loss(W2)
        L22, W22, b22 = add_layer(L2, n_hidden, 1, activation_function = None, bias=True, relu_unit="2-2")
        out2 = tf.nn.dropout(L22, keep_prob2)
        loss2 = tf.reduce_mean(tf.square(y_2 - out2) + beta * regularizers2)
        train_step2 = tf.train.AdamOptimizer(learning_rate).minimize(loss2)
        prev[1] = [sess, train_step2, x2, y_2, keep_prob2, out2, loss2]
    tf.global_variables_initializer().run()
    return run(datafile, prev, goal=goal, test_error=test_error,
               hp=hp, grid_size=grid_size, plot=plot, epochs=epochs, beta=beta, val_interval = val_interval, train_frac=train_frac,
               val_frac=val_frac, n_hidden=n_hidden, k_prob=k_prob, normalize=normalize, intervals=intervals,
               nan_ratio=nan_ratio, factor=factor, cross_validation=cross_validation, save = save)

# =============================================================================
# Runs through a series of training runs
# =============================================================================
def run(datafile, prev, test_error=False, goal='oil', grid_size = 8, plot = False, factor = 1.5, cross_validation = None,
        epochs = 1000, beta = 0.01,  val_interval = None, train_frac = 1.0, val_frac = 0.0, n_hidden = 10,
        k_prob = 1.0, normalize = True, intervals = 20, nan_ratio = 0.3, hp=1, save = False):
    df = cl.load("welltests_new.csv")
    dict_data, means, stds = cl.gen_targets(df, datafile+"", goal=goal, normalize=normalize, intervals=intervals,
                               factor = factor, nan_ratio = nan_ratio, hp=hp) #,intervals=100
    data = tens.convert_from_dict_to_tflists(dict_data)
    is_3d = False
    if (len(data[0][0]) >= 2):
        is_3d = True
        sess, train_step, x, y_, keep_prob, out, regloss = prev[1][0], prev[1][1], prev[1][2], prev[1][3], prev[1][4], prev[1][5], prev[1][6]
        print("Well",datafile, goal, "- Choke and gaslift")
    else:
        sess, train_step, x, y_, keep_prob, out, regloss = prev[0][0], prev[0][1], prev[0][2], prev[0][3], prev[0][4], prev[0][5], prev[0][6]
        print("Well",datafile, goal, "- Gaslift only")
    loss = regloss
#    loss = tf.reduce_mean(tf.abs(y_ - out))
    all_data_points = data.copy()
    val_loss = []
    train_loss = []
   

    if (cross_validation == None):
        train_set, validation_set, test_set = tens.generate_sets(data, train_frac, val_frac)
        for i in range(epochs + 1):
            batch_xs, batch_ys = tens.next_batch(train_set, 20)
#            print (batch_xs, batch_ys)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: k_prob})
            if (val_interval != None and val_frac > 0 and i%val_interval == 0):
                val_xs, val_ys = tens.next_batch(validation_set, len(validation_set))
#                print(val_xs, val_ys)
                vloss=sess.run(loss, feed_dict={x: val_xs, y_: val_ys, keep_prob: 1.0})
                val_loss.append(vloss)
                tloss = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                train_loss.append(tloss)
                print ("Step %04d" %i, " validation loss = %g" %vloss, "  training loss = %g" %tloss)


        
    if (test_error):
        y_vals = [[0] for i in range(len(test_set))]
        test_xs, test_ys = tens.next_batch(test_set, len(test_set))
        pred = sess.run(out, feed_dict={x: test_xs, y_: y_vals, keep_prob: 1.0})
        error = 0
        for i in range(len(test_xs)):
            error+=abs(pred[i][0]-test_ys[i][0])
        return error, prev, means, stds
        
    total_x, total_y = tens.total_batch(all_data_points)
    print(total_x)
    print(total_y)
    pred = sess.run(out, feed_dict={x: total_x, y_: total_y, keep_prob: 1.0})
    R2 = tens.r2(total_y, pred)
    print("R2 =",R2)
    print("Predictions:", pred)
    breakpoints = 0
    if (is_3d):
        x_vals = tens.get_x_vals(dict_data, grid_size)
        y_vals = [[0] for i in range(len(x_vals))]
        pred = sess.run(out, feed_dict={x: x_vals, y_: y_vals, keep_prob: 1.0})
        if (normalize):
            x_vals, y_vals, pred = tens.denormalize(x_vals, y_vals, pred, means, stds)
        x1 = [x[0] for x in x_vals]
        x2 = [x[1] for x in x_vals]
        z = []
        for prediction in pred:
            z.append(prediction[0])
        if (plot):
            plotter.plot3d(x1, x2, z, datafile)
        breakpoints = tools.delaunay(x1,x2,z)        
        
    else:
        if (normalize):
            total_x, total_y, pred = tens.denormalize(total_x, total_y, pred, means, stds)
        xvalues, yvalues = [], []
        for i in range(len(total_x)):
            xvalues.append(total_x[i][0])
            yvalues.append(pred[i][0])
        breakpoints = [[xvalues[0],yvalues[0]]]
        breakpoints_y = [yvalues[0]]
        breakpoints_x = [xvalues[0]]
        old_w = (yvalues[1]-yvalues[0])/(xvalues[1]-xvalues[0])
        for i in range(2,len(yvalues)):
            w = (yvalues[i]-yvalues[i-1])/(xvalues[i]-xvalues[i-1])
            if (abs(w-old_w)>0.00001):
                breakpoints.append([xvalues[i-1],yvalues[i-1]])
                breakpoints_y.append(yvalues[i-1])
                breakpoints_x.append(xvalues[i-1])
            old_w = w
        breakpoints.append([xvalues[-1],yvalues[-1]])
        breakpoints_y.append(yvalues[-1])
        breakpoints_x.append(xvalues[-1])
        
        if (plot):
            tens.plot_pred(total_x, pred, total_y)
            pyplot.ylabel(goal)
            pyplot.xlabel('gas lift')
            pyplot.title(datafile)
            pyplot.plot(breakpoints_x, breakpoints_y, 'k*')
            pyplot.show()

        ##weights, biases = sess.run(W), sess.run(b)
    if (is_3d):
        prev[1] = [sess, train_step, x, y_, keep_prob, out, loss]
    else:
        prev[0] = [sess, train_step, x, y_, keep_prob, out, loss]
    
    if (save):
        tens.save_variables(datafile, hp, goal, is_3d)
    sess.close()
    return is_3d, breakpoints, prev, total_x, pred, total_y, R2, dict_data, means, stds, train_loss, val_loss
    

    
def load(well, phase, separator ):
    filename = "" + well + "-" + separator + "-" + phase + ".txt"
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    dim = int(content[0])
    w = []
    b = []
    for k in range(1,3):
        b.append([float(x) for x in content[k+2].split()])
        if(dim == 1):
            if (k==1):
                w.append([[float(x) for x in content[k].split()]])
            else:
                w.append([float(x) for x in content[k].split()])
        else:
            content[k]=content[k].split()
            if (k==1):
                w.append([[float(content[k][x]) for x in range(len(content[k])//2)],
                       [float(content[k][x]) for x in range(len(content[k])//2,len(content[k]))]])
            else:
                w.append([float(x) for x in content[k]])
    return dim, w, b
