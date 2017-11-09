import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

def plot3d(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    print(x.shape, y.shape, z.shape)

##    x = [0,1,2,1,1,1]
##    y = [2,1,0,5,4,2]
##    z = [1, 1.5, 2,5,3,4]
    ax.set_xlabel('gas lift')
    ax.set_ylabel('choke')
    ax.set_zlabel('output')
#    ax.scatter(x, y, z)
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

##    surf = ax.plot_surface(x,y,z, cmap=cm.coolwarm,
##                       linewidth=0, antialiased=False)
    plt.show()

#plot3d(0,0,0)

