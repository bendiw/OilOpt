from keras.models import Sequential
from keras.layers import MaxoutDense, Activation
import numpy as np
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load pima indians dataset
dataset = [[0,0],[1,1],[2,4],[3,9],[4,16],[5,25],[1.1, 1.1**2], [1.5,1.5**2], [2.15, 2.15**2]]
# split into input (X) and output (Y) variables
X, Y = [], []
for i in range(len(dataset)):
    X.append(dataset[i][0])
    Y.append(dataset[i][1])
n = 3
print (X)
print (Y)
w = np.random.uniform(-0.1,0.1, size = (1,n))
b = np.random.uniform(-0.1,0.1, size = (n,))
print (w)
print (b)
print ([w,b])
## create model
##keras.layers.core.MaxoutDense(output_dim, nb_feature=1, init='glorot_uniform', weights=[w,b] , W_regularizer=None, b_regularizer=None,
##                              activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=1)
model = Sequential()
model.add(MaxoutDense(n, nb_feature=1, input_dim=1, init='uniform', weights =[w,b]))
# Compile model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10,  verbose=2)
# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
