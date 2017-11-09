import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import plotly.plotly as py
from plotly.graph_objs import *
from plotly import tools as tls
from scipy.spatial import Delaunay
import matplotlib.tri as mtri


tls.set_credentials_file(username='bendiw', api_key='qltFJLA7CmuxPoM8CBY8')

def plot3d(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    data = []
    for i in range(len(x)):
        data.append([x[i], y[i], z[i]])
    data = np.array(data)
    ax.set_xlabel('gas lift')
    ax.set_ylabel('choke')
    ax.set_zlabel('output')
    triang = mtri.Triangulation(x, y)
##    print(data[triang.triangles])

    ax.plot_trisurf(triang, z,linewidth=0.2, antialiased=True, cmap=plt.cm.CMRmap)
    plt.show()

#plot3d(0,0,0)

def delaunay(x, y, z):
    data = []
    for i in range(len(x)):
        data.append([x[i], y[i], z[i]])
    data = np.array(data)
    tri = Delaunay(data)
    print((tri.simplices[0]))
    print(data[tri.simplices])
