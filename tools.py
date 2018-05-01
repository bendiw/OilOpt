import numpy as np
import matplotlib.tri as mtri
import pandas as pd
import scipy.stats as ss

# =============================================================================
# Case 1
# =============================================================================
wellnames = ["A2", "A3", "A5", "A6", "A7", "A8", "B1", "B2", "B3","B4", "B5", "B6", "B7", "C1", "C2", "C3", "C4"] #
well_to_sep = {"A2" : ["HP"], "A3": ["HP"], "A5": ["HP"], "A6": ["HP"], "A7": ["HP"], "A8": ["HP"], 
               "B1" : ["HP", "LP"], "B2" : ["HP", "LP"], "B3" : ["HP", "LP"], "B4" : ["HP", "LP"], "B5" : ["HP", "LP"], "B6" : ["HP", "LP"], "B7" : ["HP", "LP"], 
               "C1" : ["LP"], "C2" : ["LP"], "C3" : ["LP"], "C4" : ["LP"]}
platforms = ["A", "B", "C"]
p_dict = {"A" : ["A2", "A3", "A5", "A6", "A7", "A8"], "B":["B1", "B2", 
             "B3", "B4", "B5", "B6", "B7"], "C":["C1", "C2", "C3", "C4"]}
p_sep_names = {"A":["HP"], "B":["LP", "HP"], "C":["LP"]}

# =============================================================================
# Case 2
# =============================================================================
wellnames_2= ["W"+str(x) for x in range(1,8)]
well_to_sep_2 = {w:["HP"] for w in wellnames_2}
MOP_res_columns = ["alpha", "tot_oil", "tot_gas"]+[w+"_choke" for w in wellnames_2]+[w+"_gas_mean" for w in wellnames_2]+[w+"_oil_mean" for w in wellnames_2]+[w+"_oil_var" for w in wellnames_2]+[w+"_gas_var" for w in wellnames_2]
<<<<<<< HEAD

=======
robust_res_columns = ["tot_oil", "tot_gas"]+[w+"_choke" for w in wellnames_2]+[w+"_gas_mean" for w in wellnames_2]+[w+"_oil_mean" for w in wellnames_2]
>>>>>>> master

phasenames = ["oil", "gas"]
param_dict = {'dropout':[x for x in np.arange(0.05,0.4,0.05)], 'regu':[1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]}
 
#param_dict = {'dropout':[0.1, 0.05], 'regu':[1e-6]}

#param_dict = {'dropout':[x for x in np.arange(0.1, 0.2, 0.1)], 'tau':[x for x in np.arange(1e-5, 2e-5, 1e-5)], 'length_scale':[x for x in np.arange(0.01, 0.02, 0.01)]}

# =============================================================================
# Converts dictionaries to the correctly dimensioned lists used for TF training
# =============================================================================
def convert_from_dict_to_tflists(dict_data):
    data = []
    for value in dict_data["gaslift"]:
        data.append([[value]])
    if ("choke" in dict_data.keys()):
        i = 0
        for value in dict_data["choke"]:
            data[i][0].append(value)
            i += 1
    i = 0
    for value in dict_data["output"]:
        data[i].append([value])
        i += 1
    return data


def get_limits(target, wellnames, well_to_sep, case):
    if(case==1):
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
    else:
        df = pd.read_csv("dataset_case2\\case2-oil-dataset.csv", delimiter=",", header=0)
        lower = {well:{} for well in wellnames_2}
        upper = {well:{} for well in wellnames_2}
        for well in wellnames:
            for sep in well_to_sep_2[well]:
                dfw = df[well+"_CHK_mea"]
                lower[well][sep] = dfw.min()
                upper[well][sep] = dfw.max()
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

def load(well, phase, separator, old=True, case=1):
    if(case==2):
        separator=""
    filename = "" + well + "-" + separator + "-" + phase + ".txt"
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
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

def load_2(well, phase, separator="HP", case=1, mode = "mean"):
    if(case==2):
        separator=mode
    filename = "weights/" + well + "-" + separator + "-" + phase + ".txt"
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    dims = [int(x) for x in content[0].split()]
    
# =============================================================================
#     Loads weights
# =============================================================================
    counter = 1
    w = []
    for i in range(len(dims)-1):
        w_add = []
        for k in range(int(dims[i])):
            w_add.append([float(x) for x in content[counter].split()])
            counter += 1
        w.append(w_add)
    
# =============================================================================
#     Loads bias
# =============================================================================
    b = []
    for i in range(-len(dims)+1,0):
        b.append([float(x) for x in content[i].split()])
    
    return dims, w, b

def save_variables(datafile, hp=1, goal="oil", is_3d=False, neural=None,
                   case=1, mode="mean"):
    dims = []
    for i in range(0,len(neural),2):
        dims.append(len(neural[i]))
    dims.append(len(neural[-1]))
    if (case == 2):
        sep = mode
    elif(hp==1):
        sep = "HP"
    else:
        sep = "LP"
    filename = "weights/" + datafile + "-" + sep + "-" + goal
#    print("Filename:", filename)
    file = open(filename + ".txt", "w")
    line = ""
    for dim in dims:
        line += str(dim) + " "
    file.write(line + "\n")
    for i in range(0, 1+2*(len(dims)-2), 2):
        w = neural[i]
        for x in w:
            line = ""
            for y in x:
                line += str(y) + " "
            file.write(line+"\n")
    for i in range(1, 2+2*(len(dims)-2), 2):
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

def generate_scenario_trunc_normal(case, num_scen, sep="HP", phase="gas", lower=-4, upper=4):
    if (case == 1):
        return
    scen = ss.truncnorm.rvs(lower, upper, size=(num_scen,len(wellnames_2)))
#    df = pd.DataFrame(pd.Series(scen), dtype=wellnames_2)
    df = pd.DataFrame(columns=wellnames_2)
    for i in range(num_scen):
        df.loc[i] = scen[i]
    filename = "scenarios\case"+str(case)+"_"+phase+"_"+str(num_scen)+"_"+str(lower)+"_"+str(upper)+".csv"
    with open(filename, 'w') as f:
        df.to_csv(f,sep=";", index=False)
        
def load_scenario(case, num_scen, lower, upper, phase, sep):
    filename = "scenarios\case"+str(case)+"_"+phase+"_"+str(num_scen)+"_"+str(lower)+"_"+str(upper)+".csv"
    df = pd.read_csv(filename, sep=';')
    return df

def get_scenario(case, num_scen, lower=-4, upper=4, phase="gas", sep="HP"):
    try:
        return load_scenario(case, num_scen, lower, upper, phase, sep)
    except Exception as e:
        generate_scenario_trunc_normal(case, num_scen, sep=sep, phase=phase, lower=lower, upper=upper)
        return load_scenario(case, num_scen, lower, upper, phase, sep)
    
