import tensorflow as tf
import numpy as np
import random as r
from maxout import max_out
from matplotlib import pyplot
import caseloader as cl
import tools

def add_layer(inputs, input_size, output_size, activation_function = None):
<<<<<<< HEAD
    W = tf.Variable(np.random.uniform(-0.1, 0.1, size = (input_size, output_size)), trainable = True)
    b = tf.Variable(np.random.uniform(-0.1, 0.1, size =output_size), trainable = True)
=======
    W = tf.Variable(np.random.uniform(-1, 1, size = (input_size, output_size)), trainable = True)
    b = tf.Variable(np.random.uniform(-1, 1, size =output_size), trainable = True)
>>>>>>> refs/remotes/origin/master
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
    pyplot.plot(x,pred,'r')
    pyplot.plot(x,y,'b.')
    pyplot.show()

<<<<<<< HEAD
##df = cl.load("C:\\Users\\agmal_000\\Skole\\Optimering fordypning\\OilOpt\\OilOpt\\welltests.csv")
##data = [cl.gen_targets(df, "A6",100)]
##data.sort()
###print(data[0][1])
##data = cl.conv_to_batch(data)
##print(len(data))
##print((data))

X, Y = [], []
data = []
cases = 30
for i in range(cases):
    x = r.uniform(-10,10)
    X.append(x)
X.sort()
for x in X[:cases-5]:
    y = x**2 + r.uniform(-abs(x),abs(x))
    Y.append(y)
    data.append([[x],[y]])
for x in X[cases-5:]:
    y=x/2
    Y.append(y)
    data.append([[x],[y]])
=======
df = cl.load("C:\\Users\\Bendik\\Documents\\GitHub\\OilOpt\\welltests.csv")
data = [cl.gen_targets(df, "A5", normalize=True, intervals=20)] #,intervals=100
data = cl.conv_to_batch(data)
data.sort()
##print(len(data))
print((data))

##X, Y = [], []
##data = []
##cases = 20
##for i in range(cases):
##    x = r.uniform(-10,10)
##    X.append(x)
##X.sort()
##for x in X:
##    y = x**2 + r.uniform(-abs(x),abs(x))
##    Y.append(y)
##    data.append([[x],[y]])
>>>>>>> refs/remotes/origin/master

x = tf.placeholder(tf.float64, [None,1])
y_ = tf.placeholder(tf.float64, [None,1])

n_hidden = 5

L1, W1, b1 = add_layer(x, 1, n_hidden, activation_function = None)
L2, W2, b2 = add_layer(x, 1, n_hidden, activation_function = None)

out1 = max_out(L1, 1)
out2 = max_out(L2, 1)
out = tf.subtract(out1,out2)

error = tf.reduce_sum((tf.square(y_ - out)))
##error = tf.losses.sigmoid_cross_entropy()
<<<<<<< HEAD
train_step = tf.train.AdamOptimizer(0.05).minimize(error)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(20000 + 1):
    batch_xs, batch_ys = next_batch(data, len(data))
=======
train_step = tf.train.AdamOptimizer(0.0001).minimize(error)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(25000 + 1):
    batch_xs, batch_ys = next_batch(data, len(data)-1)
>>>>>>> refs/remotes/origin/master
##    print (batch_xs, batch_ys)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if (i % 200 == 0):
        total_x, total_y = next_batch(data, len(data)-1)
        res = sess.run(error, feed_dict={x: total_x, y_: total_y})
        print ("Step %04d" %i, " error = %g" %res)
        if(res<0.0):
            break

total_x, total_y = total_batch(data)
pred = sess.run(out, feed_dict={x: total_x, y_: total_y})
plot_pred(total_x, pred, total_y)
##weights, biases = sess.run(W), sess.run(b)
##print (weights)
##print (biases)



    
