import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.tri as mtri




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

