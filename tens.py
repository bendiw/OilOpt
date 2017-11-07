import tensorflow as tf
import numpy as np
import random as r
from maxout import max_out
from matplotlib import pyplot
import caseloader as cl
import tools

def add_layer(inputs, input_size, output_size, activation_function = None):
    W = tf.Variable(np.random.uniform(-1, 1, size = (input_size, output_size)), trainable = True)
    b = tf.Variable(np.random.uniform(-1, 1, size =output_size), trainable = True)
    output = tf.matmul(inputs, W) + b
    if activation_function is not None:
        output = activation_function(output)
        "With activation function", activation_function
    return output, W, b

def next_batch(data, size):
    batch_x, batch_y = [], []
    while len(batch_x) < size:
        ran = r.randint(0, len(data)-1)
        batch_x.append(data[ran][0])
        batch_y.append(data[ran][1])
    return np.array(batch_x), np.array(batch_y)

def total_batch(data):
    batch_x, batch_y = [], []
    for i in range(len(data)):
        batch_x.append(data[i][0])
        batch_y.append(data[i][1])
    return np.array(batch_x), np.array(batch_y)

def plot_pred(x, pred, y):
    pyplot.figure()
    bp = pyplot.plot(x,pred,'r')
    pyplot.plot(x,y,'b.')
    return bp

def generate_sets(data, train_frac, val_frac):
##    print("Size data: ", len(data))
    train_set, val_set = [], []
    train_size = int(np.round(train_frac * len(data)))
    val_size = int(np.round(val_frac * len(data)))
    test_size = len(data) - train_size - val_size
    while (len(train_set) < train_size):
        train_set.append(data.pop(r.randint(0, len(data)-1)))
    while (len(val_set) < val_size):
        val_set.append(data.pop(r.randint(0, len(data)-1)))
##    print("Train: ", len(train_set))
##    print("Val: ", len(val_set))
##    print("Test: ", len(data))
    return train_set, val_set, data

def generate_cross_sets(data, cross_validation):
    size = len(data)//cross_validation
    rest = len(data)%cross_validation
    sets = []
    for i in range(rest):
        new_set = []
        for j in range(size+1):
            new_set.append(data.pop(r.randint(0,len(data)-1)))
        sets.append(new_set)
    for i in range(cross_validation - rest):
        new_set = []
        for j in range(size):
            new_set.append(data.pop(r.randint(0,len(data)-1)))
        sets.append(new_set)
    return sets
    
def run(datafile, cross_validation = None, epochs = 5000, beta = 0.01, train_frac = 0.8, val_frac = 0.1, n_hidden = 3, k_prob = 1.0, normalize = True, intervals = 100, mode = 'new'):
    df = cl.load("welltests.csv")
    data = [cl.gen_targets(df, datafile+"", normalize=True, intervals = 100)] #,intervals=100
    data = cl.conv_to_batch(data)
    all_data_points = data.copy()

    x = tf.placeholder(tf.float64, [None,1])
    y_ = tf.placeholder(tf.float64, [None,1])
    keep_prob = tf.placeholder(tf.float64)

    L1, W1, b1 = add_layer(x, 1, n_hidden, activation_function = None)
    L2, W2, b2 = add_layer(x, 1, n_hidden, activation_function = None)
    regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)

    dropout1 = tf.nn.dropout(L1, keep_prob)
    dropout2 = tf.nn.dropout(L2, keep_prob)
    out1 = max_out(dropout1, 1)
    out2 = max_out(dropout2, 1)
    out = tf.subtract(out1,out2)

    loss = tf.reduce_mean(tf.square(y_ - out))
    loss = tf.reduce_mean(loss + beta * regularizers)
    ##error = tf.losses.sigmoid_cross_entropy()

    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    if (cross_validation == None):
        train_set, validation_set, test_set = generate_sets(data, train_frac, val_frac)
        for i in range(epochs + 1):
            batch_xs, batch_ys = next_batch(train_set, len(train_set))
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: k_prob})
            if (i % 200 == 0):
                total_x, total_y = next_batch(validation_set, len(validation_set))
                res = sess.run(loss, feed_dict={x: total_x, y_: total_y, keep_prob: 1.0})
                print ("Step %04d" %i, " validation loss = %g" %res)
        test_x, test_y = total_batch(test_set)
        test_error = sess.run(loss, feed_dict={x: test_x, y_: test_y, keep_prob: 1.0})
        print ("Test set error: ", test_error)
    else:
        print ("Using",cross_validation+"-fold cross validation")
        total_error = 0
        sets = generate_cross_sets(data, cross_validation)
        for i in range(cross_validation):
            test_set = sets[i]
            train_set = []
            for j in range(i):
                for case in sets[j]:
                    train_set.append(case)
            if (i+1 < cross_validation):
                for j in range(i+1, cross_validation):
                    for case in sets[j]:
                        train_set.append(case)
            for j in range(epochs + 1):
                batch_xs, batch_ys = next_batch(train_set, len(train_set))
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: k_prob})
                if (j == epochs):
                    test_x, test_y = next_batch(test_set, len(test_set))
                    res = sess.run(loss, feed_dict={x: test_x, y_: test_y, keep_prob: 1.0})
                    print ("Cross set run %03d" %i, "test loss = %g" %res)
                    total_error += res
        print ("Average loss: ", float(total_error)/float(cross_validation))
        
    total_x, total_y = total_batch(all_data_points)
    pred = sess.run(out, feed_dict={x: total_x, y_: total_y, keep_prob: 1.0})
    bp = plot_pred(total_x, pred, total_y)

    xvalues = bp[0].get_xdata()
    yvalues = bp[0].get_ydata()
    ##breakpoints_y = [yvalues[0]]
    ##breakpoints_x = [xvalues[0]]
    breakpoints_y = []
    breakpoints_x = []
    old_w = (yvalues[1]-yvalues[0])/(xvalues[1]-xvalues[0])
    for i in range(2,len(yvalues)):
        w = (yvalues[i]-yvalues[i-1])/(xvalues[i]-xvalues[i-1])
        if (abs(w-old_w)>0.00001):
            breakpoints_y.append(yvalues[i-1])
            breakpoints_x.append(xvalues[i-1])
        old_w = w
    ##breakpoints_y.append(yvalues[-1])
    ##breakpoints_x.append(xvalues[-1])

    pyplot.plot(breakpoints_x, breakpoints_y, 'k*')
    sess.close()
    pyplot.show()
    ##weights, biases = sess.run(W), sess.run(b)
    ##print (weights)
    ##print (biases)
            





        
