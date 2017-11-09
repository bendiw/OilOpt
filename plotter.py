import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3d(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

##    x = [0,1,2]
##    y = [2,1,0]
##    z = [1, 1.5, 2]
    ax.scatter(x, y, z)
    plt.show()
