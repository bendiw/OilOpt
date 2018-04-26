import numpy as np
import matplotlib.tri as mtri
import pandas as pd

wellnames = ["A2", "A3", "A5", "A6", "A7", "A8", "B1", "B2", "B3","B4", "B5", "B6", "B7", "C1", "C2", "C3", "C4"
             ] #
wellnames_2= ["W"+str(x) for x in range(1,8)]
well_to_sep = {"A2" : ["HP"], "A3": ["HP"], "A5": ["HP"], "A6": ["HP"], "A7": ["HP"], "A8": ["HP"], 
               "B1" : ["HP", "LP"], "B2" : ["HP", "LP"], "B3" : ["HP", "LP"], "B4" : ["HP", "LP"], "B5" : ["HP", "LP"], "B6" : ["HP", "LP"], "B7" : ["HP", "LP"], 
               "C1" : ["LP"], "C2" : ["LP"], "C3" : ["LP"], "C4" : ["LP"]}
phasenames = ["oil", "gas"]
platforms = ["A", "B", "C"]
p_dict = {"A" : ["A2", "A3", "A5", "A6", "A7", "A8"], "B":["B1", "B2", 
             "B3", "B4", "B5", "B6", "B7"], "C":["C1", "C2", "C3", "C4"]}
p_sep_names = {"A":["HP"], "B":["LP", "HP"], "C":["LP"]}



param_dict = {'dropout':[x for x in np.arange(0.05,0.4,0.05)], 'regu':[1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]}
#param_dict = {'dropout':[0.1, 0.05], 'regu':[1e-6]}

#param_dict = {'dropout':[x for x in np.arange(0.1, 0.2, 0.1)], 'tau':[x for x in np.arange(1e-5, 2e-5, 1e-5)], 'length_scale':[x for x in np.arange(0.01, 0.02, 0.01)]}


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

def load(well, phase, separator ):
    filename = "" + well + "-" + separator + "-" + phase + ".txt"
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
#    print (content)
    dim = int(content[0])
    w = []
    b = []
    for k in range(1,3):
        b.append([float(x) for x in content[k+2].split()])
        if(dim == 1):

            if (k==1):
                w.append([[float(x) for x in content[k].split()]])
            else:
                w.append([float(x) for x in content[k].split()])

        else:
            content[k]=content[k].split()
            if (k==1):
                w.append([[float(content[k][x]) for x in range(len(content[k])//2)],
                       [float(content[k][x]) for x in range(len(content[k])//2,len(content[k]))]])
            else:
                w.append([float(x) for x in content[k]])
    return dim, w, b

def save_variables(datafile, hp, goal, is_3d, neural):
    if(hp==1):
        sep = "HP"
    else:
        sep = "LP"
    filename = "" + datafile + "-" + sep + "-" + goal
#    print("Filename:", filename)
    file = open(filename + ".txt", "w")
    if (is_3d):
        print("HER")
        file.write("2\n")
    else:
        file.write("1\n")
    for i in range(0,3,2):
        line = ""
        w = neural[i]
        for x in w:
            for y in x:
                line += str(y) + " "
        file.write(line+"\n")
    for i in range(1,4,2):
        line = ""
        b = neural[i]
        for x in b:
            line += str(x) + " "
        file.write(line+"\n")
    file.close()

def get_big_M(well, phase, sep):
    M =[]
    with open(well+"_"+phase+"_"+sep+"_bigM.txt", 'r') as f:
        lines = f.readlines()
        M = [line.strip().split(", ") for line in lines]
        M = [[float(l) for l in line] for line in M]
    return M        

def get_stats(data):

    return mean, stdev

def denormalize(data, mean, stdev):    
    X = [(x*X_std[0]+mean[0]) for x in X]
    y = [(y*y_std[1]+mean[1]) for y in y]
    return X, y

def simple_denorm(data, mean, stdev):
    denorm = [x*stdev+mean for x in data]
    return denorm

def simple_node_merge(X, y, x_intervals=40, y_intervals=40):
    x_min = np.min(X)
    x_max = np.max(X)
    y_min = np.min(y)
    y_max = np.max(y)
    x_step = (x_max - x_min)/float(x_intervals)
    y_step = (y_max - y_min)/float(y_intervals)
    for i in np.arange(x_min, x_max, x_step):
        for j in np.arange(y_min, y_max, y_step):
            remove_index = []
            total_x = 0
            total_y = 0
            for k in range(len(X)):
                test_x = X[k][0]
                test_y = y[k][0]
                if(i <= test_x <= i+x_step and j <= test_y <= j+y_step):
                    total_x += test_x
                    total_y += test_y
                    remove_index.append(k)
            if (len(remove_index) > 1):
                new_x = total_x/float(len(remove_index))
                new_y = total_y/float(len(remove_index))
                remove_index.reverse()
                for k in remove_index:
                    if (k < len(X)-1):
                        X = np.append(X[:k],X[k+1:])
                        y = np.append(y[:k],y[k+1:])
                    else:
                        X = X[:k]
                        y = y[:k]
                X = np.reshape(np.append(X, new_x), [-1,1])
                y = np.reshape(np.append(y, new_y), [-1,1])
    return X,y


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
