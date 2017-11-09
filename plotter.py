import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot3d(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

##    x = [0,1,2]
##    y = [2,1,0]
##    z = [1, 1.5, 2]
    ax.scatter(x, y, z)
    plt.show()

def test3d(d_dict):
    t_x = []
    t_y = []
    t_z = []
    x, y, z = d_dict['gaslift'], d_dict['choke'], d_dict['output']
##        print(x, y, z)
    x_v = np.arange(np.nanmin(x), np.nanmax(x), (np.nanmax(x)-np.nanmin(x))/20)
    y_v = np.arange(np.nanmin(y), np.nanmax(y), (np.nanmax(y)-np.nanmin(y))/20)
    print(np.nanmax(y))
    for i in x_v:
        for j in y_v:
            t_x.append(i)
            t_y.append(j)
            t_z.append(self.model.predict([[i, j]]))
    plot3d(t_x, t_y, t_z)
