import numpy as np
import matplotlib.tri as mtri
import pandas as pd


def get_limits(target, wellnames, well_to_sep):
    df = pd.read_csv("welltests_new.csv", delimiter=",", header=0)
    lower = {well:{} for well in wellnames}
    upper = {well:{} for well in wellnames}
    for well in wellnames:
        for sep in well_to_sep[well]:
            dfw = df.loc[df["well"]==well]
            if(dfw["prs_dns"].isnull().sum()/dfw.shape[0] >= 0.7):
                lower[well][sep] = dfw[target].min()
                upper[well][sep] = dfw[target].max()
            elif(sep=="HP"):
                lower[well][sep] = dfw.loc[dfw["prs_dns"]>= 18.5][target].min()
                upper[well][sep] = dfw.loc[dfw["prs_dns"]>= 18.5][target].max()
            else:
                lower[well][sep] = dfw.loc[dfw["prs_dns"]< 18.5][target].min()
                upper[well][sep] = dfw.loc[dfw["prs_dns"]< 18.5][target].max()
    return lower, upper

def normalize(data):
    X = np.array([d for d in data[0][0]])
    y = np.array([d for d in data[0][1]])
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    X_std = np.std(X)
    y_std = np.std(y)

    X = [(x-X_mean)/X_std for x in X]
    y = [(y-y_mean)/y_std for y in y]
    return X,y

def get_stats(data):

    return mean, stdev

def denormalize(data, mean, stdev):    
    X = [(x*X_std[0]+mean[0]) for x in X]
    y = [(y*y_std[1]+mean[1]) for y in y]
    return X, y

def simple_denorm(data, mean, stdev):
    denorm = [x*stdev+mean for x in data]
    return denorm

def delaunay(x, y, z):
    
    data = []
    for i in range(len(x)):
        data.append([x[i], y[i], z[i]])
    data = np.array(data)
    triang = mtri.Triangulation(x, y)
#    print(data[triang.triangles])
#    tri = Delaunay(data)
#    return(data[tri.simplices])
    return data[triang.triangles]
