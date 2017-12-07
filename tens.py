import tensorflow as tf
import numpy as np
import random as r
from maxout import max_out
from matplotlib import pyplot
import caseloader as cl
import tools
import plotter

def add_layer(inputs, input_size, output_size, activation_function = None):
    W = tf.Variable(np.random.uniform(-1, 1, size = (input_size, output_size)), trainable = True)
    b = tf.Variable(np.random.uniform(-1, 1, size =output_size), trainable = True)
# =============================================================================
#     print("inputs",inputs.shape)
#     print("W",W.shape)
#     print("b",b.shape)
# =============================================================================
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
    data.sort()
    for i in range(len(data)):
        batch_x.append(data[i][0])
        batch_y.append(data[i][1])
    return np.array(batch_x), np.array(batch_y)

def plot_pred(x, pred, y):
    pyplot.figure()
    bp = pyplot.plot(x,pred,'#141494')
    orange = '#009292'
    a = pyplot.plot(x,y,color=orange,linestyle='None', marker = '.',markersize=8)
    return bp

def generate_sets(data, train_frac, val_frac):
##    print("Size data: ", len(data))
    train_set, val_set = [], []
    train_size = int(np.round(train_frac * len(data)))
    val_size = int(np.round(val_frac * len(data)))
    while (len(train_set) < train_size):
        train_set.append(data.pop(r.randint(0, len(data)-1)))
    while (len(val_set) < val_size):
        val_set.append(data.pop(r.randint(0, len(data)-1)))
##    print("Train: ", len(train_set))
##    print("Val: ", len(val_set))
##    print("Test: ", len(data))
    train_set.sort()
    val_set.sort()
    data.sort()
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

def convert_from_dict_to_tflists(dict_data):
    data = []
    for value in dict_data["gaslift"]:
        data.append([[value]])
    if ("choke" in dict_data.keys()):
        i = 0
        for value in dict_data["choke"]:
            data[i][0].append(value)
            i += 1
    i = 0
    for value in dict_data["output"]:
        data[i].append([value])
        i += 1
    return data

def get_x_vals(dict_data, grid_size):
    x1_min = np.nanmin(dict_data["gaslift"])
    x1_max = np.nanmax(dict_data["gaslift"])
    x2_min = np.nanmin(dict_data["choke"])
    x2_max = np.nanmax(dict_data["choke"])
    x1 = np.arange(x1_min, x1_max, (x1_max-x1_min)/grid_size)
    x2 = np.arange(x2_min, x2_max, (x2_max-x2_min)/grid_size)
    x_vals = []
    for i in range(len(x1)):
        for j in range(len(x2)):
            x_vals.append([x1[i],x2[j]])
    return x_vals
    
def r2(ydata, pred):
    tot = 0
    for i in range(len(ydata)):
        tot += ydata[i][0]
    mean = tot/len(ydata)
    r2 = 1 - res_ss(ydata,pred) / tot_ss(ydata,mean)
    return r2
        
def tot_ss(ys, mean):
    ss_tot = 0
    for i in range(len(ys)):
        ss_tot += (ys[i][0]-mean)**2
    return ss_tot

# =============================================================================
# def reg_ss(pred, mean):
#     ss_reg = 0
#     for i in range(len(pred)):
#         ss_reg += (pred[i][0]-mean)**2
#     return ss_reg
# =============================================================================

def res_ss(ys, pred):
    ss_res = 0
    for i in range(len(ys)):
        ss_res += (ys[i][0]-pred[i][0])**2
    return ss_res

def denormalize(x, y, pred, means, stds):
    y_ = tools.simple_denorm([a[0] for a in y], means[-1], stds[-1])
    pred_ = tools.simple_denorm([a[0] for a in pred], means[-1], stds[-1])
    new_y = [[a] for a in y_]
    new_pred = [[a] for a in pred_]
    x1 = tools.simple_denorm([a[0] for a in x], means[0], stds[0])
    if (len(x[0])>1):
        x2 = tools.simple_denorm([a[1] for a in x], means[1], stds[1])
        new_x = []
        for i in range(len(x)):
            new_x.append([x1[i],x2[i]])
    else:
        new_x = [[x] for x in x1]
    return new_x, new_y, new_pred

def run2(datafile):
    is_3d, breakpoints, total_x, pred, total_y = run(datafile, beta = 0)
    plot_pred(total_x, pred, total_y)
    pyplot.plot([x[0] for x in breakpoints], [x[1] for x in breakpoints], 'k*')
    pyplot.ylabel('oil')
    pyplot.xlabel('gaslift')
    pyplot.title(datafile)
    colors = ['b', 'g', 'c', 'm', 'y', 'k', 'w']
    c = 0
    for i in range(2,10,2):
        _, breakpoints, _, pred, _ = run(datafile, beta = float(i)/100.0)
        bp = pyplot.plot(total_x, pred, colors[c])
        pyplot.plot([x[0] for x in breakpoints], [x[1] for x in breakpoints], 'k*')
        c+=1
    pyplot.show()

def run(datafile, goal='oil', grid_size = 15, plot = False, factor = 1.5, cross_validation = None,
        epochs = 2000, beta = 0.05, train_frac = 1.0, val_frac = 0.0, n_hidden = 5,
        k_prob = 1.0, normalize = True, intervals = 20, nan_ratio = 0.3, hp=0):
    df = cl.load("welltests.csv")
    dict_data, means, stds = cl.gen_targets(df, datafile+"", goal=goal, normalize=True, intervals=intervals,
                               factor = factor, nan_ratio = nan_ratio, hp=hp) #,intervals=100
    data = convert_from_dict_to_tflists(dict_data)
    is_3d = False
    if (len(data[0][0]) >= 2):
        is_3d = True
        print("Well",datafile, goal, "- Choke and gaslift")
    else:
        print("Well",datafile, goal, "- Gaslift only")
    all_data_points = data.copy()

    num_inputs = len(data[0][0])
    x = tf.placeholder(tf.float64, [None,num_inputs])
    y_ = tf.placeholder(tf.float64, [None,1])
    keep_prob = tf.placeholder(tf.float64)
    L1, W1, b1 = add_layer(x, num_inputs, n_hidden, activation_function = None)
    L2, W2, b2 = add_layer(x, num_inputs, n_hidden, activation_function = None)        

    regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)

    dropout1 = tf.nn.dropout(L1, keep_prob)
    dropout2 = tf.nn.dropout(L2, keep_prob)
    out1 = max_out(dropout1, 1)
    out2 = max_out(dropout2, 1)
    out = tf.subtract(out1,out2)

    loss = tf.reduce_mean(tf.square(y_ - out))
    loss = tf.reduce_mean(loss + beta * regularizers)
    ##error = tf.losses.sigmoid_cross_entropy()
# =============================================================================
#     print("Start training")
# =============================================================================
    train_step = tf.train.AdamOptimizer(0.03).minimize(loss)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    if (cross_validation == None):
        train_set, validation_set, test_set = generate_sets(data, train_frac, val_frac)
        for i in range(epochs + 1):
            batch_xs, batch_ys = next_batch(train_set, len(train_set))
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: k_prob})
# =============================================================================
#             if (i % 200 == 0):
#                 total_x, total_y = next_batch(validation_set, len(validation_set))
#                 res = sess.run(loss, feed_dict={x: total_x, y_: total_y, keep_prob: 1.0})
#                 print ("Step %04d" %i, " validation loss = %g" %res)
#         test_x, test_y = total_batch(test_set)
#         test_error = sess.run(loss, feed_dict={x: test_x, y_: test_y, keep_prob: 1.0})
#         print ("Test set error: ", test_error)
# =============================================================================
# =============================================================================
#     else:
#         print ("Using ",cross_validation,"-fold cross validation")
#         total_error = 0
#         sets = generate_cross_sets(data, cross_validation)
#         for i in range(cross_validation):
#             test_set = sets[i]
#             train_set = []
#             for j in range(i):
#                 for case in sets[j]:
#                     train_set.append(case)
#             if (i+1 < cross_validation):
#                 for j in range(i+1, cross_validation):
#                     for case in sets[j]:
#                         train_set.append(case)
#             for j in range(epochs + 1):
#                 batch_xs, batch_ys = next_batch(train_set, len(train_set))
#                 sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: k_prob})
#                 if (j == epochs):
#                     test_x, test_y = next_batch(test_set, len(test_set))
#                     res = sess.run(loss, feed_dict={x: test_x, y_: test_y, keep_prob: 1.0})
#                     print ("Cross set run %03d" %i, "test loss = %g" %res)
#                     total_error += res
#         print ("Average loss: ", float(total_error)/float(cross_validation))
# =============================================================================
        
    total_x, total_y = total_batch(all_data_points)
    pred = sess.run(out, feed_dict={x: total_x, y_: total_y, keep_prob: 1.0})
    print("R2 =",r2(total_y,pred))
    breakpoints = 0
    if (is_3d):
        x_vals = get_x_vals(dict_data, grid_size)
        y_vals = [[0] for i in range(len(x_vals))]
        pred = sess.run(out, feed_dict={x: x_vals, y_: y_vals, keep_prob: 1.0})
        x_vals, y_vals, pred = denormalize(x_vals, y_vals, pred, means, stds)
        x1 = [x[0] for x in x_vals]
        x2 = [x[1] for x in x_vals]
        z = []
        for prediction in pred:
            z.append(prediction[0])
        if (plot):
            plotter.plot3d(x1, x2, z, datafile)
        breakpoints = tools.delaunay(x1,x2,z)        
        
    else:
        total_x, total_y, pred = denormalize(total_x, total_y, pred, means, stds)
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
            plot_pred(total_x, pred, total_y)
            pyplot.ylabel(goal)
            pyplot.xlabel('gas lift')
            pyplot.title(datafile)
            pyplot.plot(breakpoints_x, breakpoints_y, 'k*')
            pyplot.show()

        ##weights, biases = sess.run(W), sess.run(b)
        
    sess.close()
    return is_3d, breakpoints#, total_x, pred, total_y
    
    






        
