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

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr

def plot3d(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    data = []
    for i in range(len(x)):
        data.append([x[i], y[i], z[i]])
##    print(x.shape, y.shape, z.shape)
    data = np.array(data)
##    x = [0,1,2,1,1,1]
##    y = [2,1,0,5,4,2]
##    z = [1, 1.5, 2,5,3,4]
    ax.set_xlabel('gas lift')
    ax.set_ylabel('choke')
    ax.set_zlabel('output')
#    ax.scatter(x, y, z)
    triang = mtri.Triangulation(x, y)
    print(data[triang.triangles])

    ax.plot_trisurf(triang, z,linewidth=0.2, antialiased=True, cmap=plt.cm.CMRmap)
    plt.show()

#plot3d(0,0,0)

def delaunay(x, y, z):
    data = []
    for i in range(len(x)):
        data.append([x[i], y[i], z[i]])
    data = np.array(data)
##    print(data.shape)
    tri = Delaunay(data)
    print((tri.simplices[0]))
    print(data[tri.simplices])
##    norm = calc_norm(data, tri.simplices)
##    fac = norm[1][0]/norm[0][0]

def calc_norm(vertices, faces):
    #Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    #Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle             
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices, 
    # we need to normalize these, so that our next step weights each normal equally.
##    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle, 
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
##    norm[ faces[:,0] ] += n
##    norm[ faces[:,1] ] += n
##    norm[ faces[:,2] ] += n
##    normalize_v3(norm)
##    return norm
    return n

def mesh(x, y, z):
    points=Scatter3d(mode = 'markers',
                 name = '',
                 x =x,
                 y= y,
                 z= z,
                 marker = Marker( size=2, color='#458B00' ))
    simplexes = Mesh3d(alphahull =-1,
                   name = '',
                   x =x,
                   y= y,
                   z= z,
                   color='90EE90', #set the color of simplexes in alpha shape
                   opacity=0.15)
##    print(x, "\n")
##    print(simplexes.x)
##    return



    x_style = dict( zeroline=False, range=[np.nanmin(x), np.nanmax(x)], tickvals=np.linspace(-2.85, 4.25, 5)[1:].round(1))
    y_style = dict( zeroline=False, range=[np.nanmin(y), np.nanmax(y)], tickvals=np.linspace(-2.65, 1.32, 4)[1:].round(1))
    z_style = dict( zeroline=False, range=[np.nanmin(z), np.nanmax(z)], tickvals=np.linspace(-3.67, 1.4, 5).round(1))
    layout=Layout(title='Mesh grid',
              width=500,
              height=500,
              scene = Scene(
              xaxis = x_style,
              yaxis = y_style,
              zaxis = z_style
             ))
    fig=Figure(data=Data([points, simplexes]), layout=layout)
##    py.sign_in('bendiw', 'smtbajoo93')
    py.plot(fig, filename='3D-AlphaS-ex')
