import numpy as np
import matplotlib.tri as mtri
import pandas as pd

wellnames = ["A2", "A3", "A5", "A6", "A7", "A8", "B1", "B2", 
             "B3", "B4", "B5", "B6", "B7", "C1", "C2", "C3", "C4"]


well_to_sep = {"A2" : ["HP"], "A3": ["HP"], "A5": ["HP"], "A6": ["HP"], "A7": ["HP"], "A8": ["HP"], 
               "B1" : ["HP", "LP"], "B2" : ["HP", "LP"], "B3" : ["HP", "LP"], "B4" : ["HP", "LP"], "B5" : ["HP", "LP"], "B6" : ["HP", "LP"], "B7" : ["HP", "LP"], 
               "C1" : ["LP"], "C2" : ["LP"], "C3" : ["LP"], "C4" : ["LP"]}
phasenames = ["oil", "gas"]
platforms = ["A", "B", "C"]
p_dict = {"A" : ["A2", "A3", "A5", "A6", "A7", "A8"], "B":["B1", "B2", 
             "B3", "B4", "B5", "B6", "B7"], "C":["C1", "C2", "C3", "C4"]}
p_sep_names = {"A":["HP"], "B":["LP", "HP"], "C":["LP"]}


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
