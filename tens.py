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

df = cl.load("welltests.csv")
<<<<<<< HEAD
data = [cl.gen_targets(df, "C3", normalize=True, intervals = 100)] #,intervals=100
=======
##df = cl.load("C:\\Users\\Bendik\\Documents\\GitHub\\OilOpt\\welltests.csv")
data = [cl.gen_targets(df, "A5", normalize=True)] #,intervals=100
>>>>>>> refs/remotes/origin/master
data = cl.conv_to_batch(data)
data.sort()
##print(len(data))
##print((data))



x = tf.placeholder(tf.float64, [None,1])
y_ = tf.placeholder(tf.float64, [None,1])

n_hidden = 100

L1, W1, b1 = add_layer(x, 1, n_hidden, activation_function = None)
L2, W2, b2 = add_layer(x, 1, n_hidden, activation_function = None)

out1 = max_out(L1, 1)
out2 = max_out(L2, 1)
out = tf.subtract(out1,out2)

error = tf.reduce_mean((tf.square(y_ - out)))
##error = tf.losses.sigmoid_cross_entropy()

train_step = tf.train.AdamOptimizer(0.01).minimize(error)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(5000 + 1):
    batch_xs, batch_ys = next_batch(data, len(data))
##    print (batch_xs, batch_ys)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if (i % 200 == 0):
        total_x, total_y = next_batch(data, len(data))
        res = sess.run(error, feed_dict={x: total_x, y_: total_y})
        print ("Step %04d" %i, " error = %g" %res)
        if(res<0.05):
            break

total_x, total_y = total_batch(data)
pred = sess.run(out, feed_dict={x: total_x, y_: total_y})
bp = plot_pred(total_x, pred, total_y)

xvalues = bp[0].get_xdata()
yvalues = bp[0].get_ydata()
breakpoints_y = [yvalues[0]]
breakpoints_x = [xvalues[0]]
old_w = (yvalues[1]-yvalues[0])/(xvalues[1]-xvalues[0])
for i in range(2,len(yvalues)-1):
    w = (yvalues[i]-yvalues[i-1])/(xvalues[i]-xvalues[i-1])
    print ("")
    print ("w:",w)
    print ("old_w:",old_w)
    if (abs(w-old_w)>0.00001):
        breakpoints_y.append(yvalues[i-1])
        breakpoints_x.append(xvalues[i-1])
    old_w = w
breakpoints_y.append(yvalues[-1])
breakpoints_x.append(xvalues[-1])

print (breakpoints_x)
print (breakpoints_y)
pyplot.plot(breakpoints_x, breakpoints_y, 'k*', markersize = 6.0)
sess.close()
pyplot.show()
##weights, biases = sess.run(W), sess.run(b)
##print (weights)
##print (biases)



    
