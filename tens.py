import tensorflow as tf
import numpy as np
import random as r
from maxout import max_out
from matplotlib import pyplot
import caseloader as cl
import tools
import plotter
import math

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
    bp, = pyplot.plot(x,pred,'#141494', label = "HP")
    orange = '#009292'
    a, = pyplot.plot(x,y,color=orange,linestyle='None', marker = '.',markersize=8)
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
    x1 = np.linspace(x1_min,x1_max,num=grid_size)
    x2 = np.linspace(x2_min,x2_max,num=grid_size)
#    x1 = np.arange(x1_min, x1_max+(x1_max-x1_min)/grid_size, (x1_max-x1_min)/grid_size)
#    x2 = np.arange(x2_min, x2_max+(x2_max-x2_min)/grid_size, (x2_max-x2_min)/grid_size)
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

def betaplot(datafile, beta, n, train_frac):
    prev = None
    pyplot.figure()
    pyplot.title(datafile)
    pyplot.ylabel("oil")
    pyplot.xlabel('gas lift')
    for i in range(n):
        if(i%10==0):
            print ("Iteration",i)
        _, _, prev, x, pred, _, _, _, _, _, _, _ = hey(datafile, beta=beta, train_frac = train_frac, val_frac = 1-train_frac, 
                                                       prev=prev,nan_ratio=0.0,intervals=50)
#    print(trainnp)
#    print(valnp)
#    print(iterations)
        pyplot.plot(x, pred, '#141494', alpha=0.7)
    _, _, _, total_x, _, total_y, _, _, _, _, _, _ = hey(datafile, beta=beta, 
                                                       prev=prev,nan_ratio=0.0,intervals=50)
    pyplot.plot([x[0] for x in total_x], [x[0] for x in total_y],
                '#009292', linestyle = 'None', marker = '.',markersize=8)
    pyplot.show()

def hplp():
    wells = ["B1","B2","B3","B4","B5","B6","B7"]
    w = ["B1","B2"]
    colors = ['r','b', 'g', 'c', 'm', 'y', 'k', 'w']
    prev = None
    for i in range(len(wells)):
        prev = hplp_run(wells[i],colors[i], prev)
#        
#    legend = pyplot.legend()
#    for label in legend.get_texts():
#        label.set_fontsize('large')
#    for label in legend.get_lines():
#        label.set_linewidth(3)  # the legend line width
    pyplot.show()
    
def find_4(point, x_coarse):
    p1=point[0]
    p2=point[1]
    lower1, higher1, lower2, higher2 = 99999999, 99999999, 99999999, 99999999
    for coarse_point in x_coarse:
        if (p1 - coarse_point[0] >= -0.001 and p1 - coarse_point[0] <= lower1):
            l1 = coarse_point[0]
            lower1 = p1 - coarse_point[0]
        if (p1 - coarse_point[0] <= 0.001 and coarse_point[0] - p1 <= higher1):
            h1 = coarse_point[0]
            higher1 = p1 - coarse_point[0]
        if (p2 - coarse_point[1] >= -0.001 and p2 - coarse_point[1] <= lower2):
            l2 = coarse_point[1]
            lower2 = p2 - coarse_point[1]
        if (p2 -coarse_point[1] <= 0.001 and coarse_point[1] - p2 <= higher2):
            h2 = coarse_point[1]
            higher2 = p2 - coarse_point[1]
    if (l1 == h1 and l2 == h2):
        return [[l1,l2]]
    elif(l1 == h1):
        return [[l1,l2], [l1,h2]]
    elif(l2 == h2):
        return [[l1,l2], [h1,l2]]
    return [[l1,l2],[l1,h2],[h1,l2],[h1,h2]]


def err_4(point, closest):
    dist = []
    s = 0.0
    for i in range(len(closest)):
        p  = math.sqrt((point[0]-closest[i][0])**2 + (point[1]-closest[i][1])**2)
        if (p>0):
            s+=1/float(p)
            dist.append(1/float(p))
    if (s > 0):
        for i in range(len(dist)):
            dist[i] = dist[i]/s
    else:
        dist=[1.0]
    return dist
        

def griderror():
    wells = ["A5","A6","B1","B4","C4"]
    grids = [2,4,7,10,15,20]
    error = {}
    for well in wells:
        error[well] = []
    prev = None
    K=5
    for well in wells:
        _, _, prev, _, _, _, _, dict_data, means, stds = hey(well, prev=prev,nan_ratio=0.6)
        for grid in grids:
            poserr_tot = 0
            negerr_tot = 0
            errs_tot = []
            for i in range(K):
                print("Well",well,"- Grid",grid, "- Run",i)
                poserr = 0
                negerr = 0
                x_vals_fine = get_x_vals(dict_data, 100)
                y_vals_fine = [[0] for i in range(len(x_vals_fine))]
                pred_fine = prev[1][0].run(prev[1][5], feed_dict={prev[1][2]: x_vals_fine, prev[1][3]: y_vals_fine, prev[1][4]: 1.0})
                x_vals_fine, y_vals_fine, pred_fine = denormalize(x_vals_fine, y_vals_fine, pred_fine, means, stds)
                
                x_vals_coarse = get_x_vals(dict_data, grid)
                y_vals_coarse = [[0] for i in range(len(x_vals_coarse))]
                pred_coarse = prev[1][0].run(prev[1][5], feed_dict={prev[1][2]: x_vals_coarse, prev[1][3]: y_vals_coarse, prev[1][4]: 1.0})
                x_vals_coarse, y_vals_coarse, pred_coarse = denormalize(x_vals_coarse, y_vals_coarse, pred_coarse, means, stds)
                errs = [0]*len(x_vals_fine)
                for p in range(len(x_vals_fine)):
                    closest = find_4(x_vals_fine[p], x_vals_coarse)
                    weights = err_4(x_vals_fine[p], closest)
                    for j in range(len(pred_coarse)):
                        for k in range(len(closest)):
                            if (x_vals_coarse[j][0] == closest[k][0] and x_vals_coarse[j][1] == closest[k][1]):
                                diff = (pred_coarse[j][0] - pred_fine[p][0])*weights[k]
                                if (diff > 0):
                                    poserr+=diff
                                else:
                                    negerr+=diff
                                errs[p] += (abs(diff))
                poserr_tot += poserr/float(K)
                negerr_tot += negerr/float(K)
                if (len(errs_tot) == 0):
                    for t in errs:
                        errs_tot.append(t/float(K))
                else:
                    for j in range(len(errs)):
                        errs_tot[j] += errs[j]/float(K)
            avg = sum(errs_tot)/float(len(errs_tot))
            std = 0
            for e in errs_tot:
                std += math.pow(e - avg, 2)
            error[well].append([avg, poserr_tot/float(len(x_vals_fine)), negerr_tot/float(len(x_vals_fine)), math.sqrt(std/float(len(errs_tot)-1))])
    return error


#                x1_coarse = [x[0] for x in x_vals_coarse]
#                x2_coarse = [x[1] for x in x_vals_coarse]
#                z_coarse = []
#                for prediction in pred_coarse:
#                    z_coarse.append(prediction[0])
#                breakpoints_coarse = tools.delaunay(x1_coarse,x2_coarse,z_coarse)
                
def lossplot(datafile, runs, epochs, train_frac,learning_rate, interval, beta = 0.1):
    train_losses = []
    val_losses = []
    final_train = []
    final_val = []
    prev = None
    valnp = 0
    trainnp = 0
    for i in range(runs):
        _, _, prev, _, _, _, _, _, _, _, train_loss, val_loss = hey(datafile,learning_rate=learning_rate,epochs=epochs, beta=beta,prev=prev, val_interval = interval, train_frac = train_frac, val_frac = 1.0-train_frac)
        train_losses.append(np.array(train_loss))
        val_losses.append(np.array(val_loss))
    for i in range(len(train_losses)):
        trainnp += train_losses[i]/float(runs)
        valnp += val_losses[i]/float(runs)
    iterations = [i for i in range(0,epochs+1,interval)]
#    print(trainnp)
#    print(valnp)
#    print(iterations)
    pyplot.figure()
    pyplot.ylabel("loss")
    pyplot.xlabel('iteration')
    pyplot.title(datafile)
    pyplot.plot(iterations, trainnp, '#141494', label="Training loss")
    pyplot.plot(iterations, valnp, '#009292', label="Validation loss")
    legend = pyplot.legend()
    pyplot.show()
        
def getmax(pred):
    m = 0
    i_ = 0
    for i in range(len(pred)):
        if (pred[i][0] > m):
            m = pred[i][0]
            i_ = i
    return i_

def hplp_run(datafile, color, prev):
    is_3d, breakpoints, prev, total_x, pred, total_y, R = hey(datafile, prev = prev, nan_ratio = 0.0, beta=0.15, epochs=1000,hp=1)
    i_ = getmax(pred)
    hp, = pyplot.plot(total_x[i_], pred[i_], color=color, linestyle = "none", marker = '^', label = "HP")
    a=pyplot.ylabel('oil')
    a=pyplot.xlabel('gaslift')
    
    _, _, prev, total_x, pred, _, R2 = hey(datafile, prev = prev, epochs=1000, nan_ratio = 0.0, beta = 0.15, hp=2)
    i_ = getmax(pred)
    lp, = pyplot.plot(total_x[i_], pred[i_], color=color, linestyle = "none", marker = 's', label = "LP")
#    pyplot.plot([x[0] for x in breakpoints], [x[1] for x in breakpoints], 'k*')

    return prev


def run2(datafile,beta):
    is_3d, breakpoints, prev, total_x, pred, total_y, R = hey(datafile, beta=beta, epochs=1000)
#    plot_pred(total_x, pred, total_y)
##    pyplot.plot([x[0] for x in breakpoints], [x[1] for x in breakpoints], 'k*')
#    pyplot.ylabel('oil')
#    pyplot.xlabel('gaslift')
#    pyplot.title(datafile)
#    colors = ['b', 'g', 'c', 'm', 'y', 'k', 'w']
#    c = 0
    for i in range(99):
        _, _, prev, _, _, _, R2 = hey(datafile, epochs=1000, prev=prev, beta=beta)
#        bp = pyplot.plot(total_x, pred, colors[c])
#        pyplot.plot([x[0] for x in breakpoints], [x[1] for x in breakpoints], 'k*')
#        c+=1
        R += R2
    print ("Average:",R/100.0)
    for p in prev:
        if (len(p)>1):
            p[0].close()
#    pyplot.show()
    
def hey(datafile, goal='oil', grid_size = 8, plot = False,learning_rate=0.03, factor = 1.5, cross_validation = None,
        epochs = 500, beta = 0.1, val_interval = None, train_frac = 1.0, val_frac = 0.0, n_hidden = 5,
        k_prob = 1.0, normalize = True, intervals = 20, nan_ratio = 0.3, hp=1, prev = None):
    if (prev == None):
        prev = [0]*2
        x = tf.placeholder(tf.float64, [None,1])
        y_ = tf.placeholder(tf.float64, [None,1])
        keep_prob = tf.placeholder(tf.float64)
        L1, W1, b1 = add_layer(x, 1, 5, activation_function = None)
        L2, W2, b2 = add_layer(x, 1, 5, activation_function = None) 
        regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
        dropout1 = tf.nn.dropout(L1, keep_prob)
        dropout2 = tf.nn.dropout(L2, keep_prob)
        out1 = max_out(dropout1, 1)
        out2 = max_out(dropout2, 1)
        out = tf.subtract(out1,out2)
        loss = tf.reduce_mean(tf.square(y_ - out) + beta * regularizers)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess = tf.InteractiveSession()
        prev[0] = [sess, train_step, x, y_, keep_prob, out, loss]
        print("Initializing Tensorflow modules. This may take a while...")
        
        x2 = tf.placeholder(tf.float64, [None,2])
        y_2 = tf.placeholder(tf.float64, [None,1])
        keep_prob2 = tf.placeholder(tf.float64)
        L12, W12, b12 = add_layer(x2, 2, 5, activation_function = None)
        L22, W22, b22 = add_layer(x2, 2, 5, activation_function = None) 
        regularizers2 = tf.nn.l2_loss(W12) + tf.nn.l2_loss(W22)
        dropout12 = tf.nn.dropout(L12, keep_prob2)
        dropout22 = tf.nn.dropout(L22, keep_prob2)
        out12 = max_out(dropout12, 1)
        out22 = max_out(dropout22, 1)
        out2 = tf.subtract(out12,out22)
        loss2 = tf.reduce_mean(tf.square(y_2 - out2) + beta * regularizers2)
        train_step2 = tf.train.AdamOptimizer(learning_rate).minimize(loss2)
        prev[1] = [sess, train_step2, x2, y_2, keep_prob2, out2, loss2]
    tf.global_variables_initializer().run()
    return run(datafile, prev, goal=goal, 
               hp=hp, grid_size=grid_size, plot=plot, epochs=epochs, beta=beta, val_interval = val_interval, train_frac=train_frac,
               val_frac=val_frac, n_hidden=n_hidden, k_prob=k_prob, normalize=normalize, intervals=intervals,
               nan_ratio=nan_ratio, factor=factor, cross_validation=cross_validation)

def run(datafile, prev, goal='oil', grid_size = 8, plot = False, factor = 1.5, cross_validation = None,
        epochs = 500, beta = 0.1,  val_interval = None, train_frac = 1.0, val_frac = 0.0, n_hidden = 5,
        k_prob = 1.0, normalize = True, intervals = 20, nan_ratio = 0.3, hp=0):
    df = cl.load("welltests_new.csv")
    dict_data, means, stds = cl.gen_targets(df, datafile+"", goal=goal, normalize=True, intervals=intervals,
                               factor = factor, nan_ratio = nan_ratio, hp=hp) #,intervals=100
    data = convert_from_dict_to_tflists(dict_data)
    is_3d = False
    if (len(data[0][0]) >= 2):
        is_3d = True
        sess, train_step, x, y_, keep_prob, out, regloss = prev[1][0], prev[1][1], prev[1][2], prev[1][3], prev[1][4], prev[1][5], prev[1][6]
        print("Well",datafile, goal, "- Choke and gaslift")
    else:
        sess, train_step, x, y_, keep_prob, out, regloss = prev[0][0], prev[0][1], prev[0][2], prev[0][3], prev[0][4], prev[0][5], prev[0][6]
        print("Well",datafile, goal, "- Gaslift only")
    loss = tf.reduce_mean(tf.square(y_ - out))
    all_data_points = data.copy()
    val_loss = []
    train_loss = []
   
# =============================================================================
#     num_inputs = len(data[0][0])
#     x = tf.placeholder(tf.float64, [None,num_inputs])
#     y_ = tf.placeholder(tf.float64, [None,1])
#     keep_prob = tf.placeholder(tf.float64)
#     L1, W1, b1 = add_layer(x, num_inputs, n_hidden, activation_function = None)
#     L2, W2, b2 = add_layer(x, num_inputs, n_hidden, activation_function = None) 
# 
#     regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
# 
#     dropout1 = tf.nn.dropout(L1, keep_prob)
#     dropout2 = tf.nn.dropout(L2, keep_prob)
#     out1 = max_out(dropout1, 1)
#     out2 = max_out(dropout2, 1)
#     out = tf.subtract(out1,out2)
# 
#     #loss = tf.reduce_mean(tf.square(y_ - out))
#     loss = tf.reduce_mean(tf.square(y_ - out) + beta * regularizers)
#     ##error = tf.losses.sigmoid_cross_entropy()
# # =============================================================================
# #     print("Start training")
# # =============================================================================
#     train_step = tf.train.AdamOptimizer(0.03).minimize(loss)
#     sess = tf.InteractiveSession()
#     tf.global_variables_initializer().run()
# =============================================================================
    if (cross_validation == None):
        train_set, validation_set, test_set = generate_sets(data, train_frac, val_frac)
        for i in range(epochs + 1):
            batch_xs, batch_ys = next_batch(train_set, len(train_set))
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: k_prob})
            if (val_interval != None and val_frac > 0 and i%val_interval == 0):
                val_xs, val_ys = next_batch(validation_set, len(validation_set))
#                print(val_xs, val_ys)
                vloss=sess.run(loss, feed_dict={x: val_xs, y_: val_ys, keep_prob: 1.0})
                val_loss.append(vloss)
                tloss = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                train_loss.append(tloss)
#                print ("Step %04d" %i, " validation loss = %g" %vloss, "  training loss = %g" %tloss)
# =============================================================================
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
    R2 = r2(total_y, pred)
    print("R2 =",R2)
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
    if (is_3d):
        prev[1] = [sess, train_step, x, y_, keep_prob, out, loss]
    else:
        prev[0] = [sess, train_step, x, y_, keep_prob, out, loss]

    return is_3d, breakpoints, prev, total_x, pred, total_y, R2, dict_data, means, stds, train_loss, val_loss
    
    






        
