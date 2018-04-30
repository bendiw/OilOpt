import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.tri as mtri


def first_plot_3d(x,y,z,x_test,y_test,z_test,well):
    fig = plt.figure()
    plt.autoscale(False)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('gas lift')
    ax.set_ylabel('choke')
    ax.set_zlabel('output')
    plt.title(well)
    plt.show()
    
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    z_test = np.array(z_test)
    data, data_test = [], []
    for i in range(len(x_test)):
        data_test.append([x_test[i], y_test[i], z_test[i]])
    data_test = np.array(data_test)

    points = ax.scatter(x,y,zs=z,c='b')
    triang = mtri.Triangulation(x_test, y_test)
    tri_plot = ax.plot_trisurf(triang, z_test,linewidth=0.2, antialiased=True, cmap=plt.cm.PuBu,alpha=0.1)
    plt.pause(0.001)
    
    return triang, ax
    
def update_3d(x,y,z,z_test, triang, ax):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    z_test = np.array(z_test)
    ax.collections.clear()
    points = ax.scatter(x,y,zs=z,c='b')
    tri_plot = ax.plot_trisurf(triang, z_test,linewidth=0.2, antialiased=True, cmap=plt.cm.PuBu, alpha=0.1)
    plt.pause(0.001)


def plot3d(x, y, z, well):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    data = []
    for i in range(len(x)):
        data.append([x[i], y[i], z[i]])
    data = np.array(data)
    plt.title(well)
    ax.set_xlabel('gas lift')
    ax.set_ylabel('choke')
    ax.set_zlabel('output')
    triang = mtri.Triangulation(x, y)
#    print(data[triang.triangles])
    #ax.plot_trisurf(triang, z,linewidth=0.2, antialiased=True)
#    delaunay(x, y, z)
    ax.plot_trisurf(triang, z,linewidth=0.2, antialiased=True, cmap=plt.cm.PuBu)
    plt.show()
    return fig

#plot3d(0,0,0)

