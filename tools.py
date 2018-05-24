import numpy as np
import matplotlib.tri as mtri
import pandas as pd
import math
import scipy.stats as ss
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from keras import backend as K
from scipy.stats import randint, uniform
from matplotlib import pyplot

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

well_GORs = {'W1': 1578.6439, 'W2': 5886.233, 'W3': 2085.9548, 'W4': 1682.6052, 'W5': 1676.8193, 'W6': 2630.8806, 'W7': 5608.433}
well_order = ["W1", "W5", "W4", "W3", "W6", "W7", "W2"]
tot_exp_caps = {"under_cap":250000, "zero":250000, "over_cap":225000, "over_cap_old":225000} #225000


# =============================================================================
# Case 2
# =============================================================================
wellnames_2= ["W"+str(x) for x in range(1,8)]
well_to_sep_2 = {w:["HP"] for w in wellnames_2}
MOP_res_columns = ["alpha", "tot_oil", "tot_gas"]+[w+"_choke" for w in wellnames_2]+[w+"_gas_mean" for w in wellnames_2]+[w+"_oil_mean" for w in wellnames_2]+[w+"_oil_var" for w in wellnames_2]+[w+"_gas_var" for w in wellnames_2]

base_res = ["scenarios", "tot_cap"]+["tot_oil", "tot_gas"]+[w+"_indiv_cap" for w in wellnames_2]+[w+"_choke" for w in wellnames_2]+[w+"_gas_mean" for w in wellnames_2]+[w+"_oil_mean" for w in wellnames_2]
robust_res_columns = base_res+[w+"_gas_var" for w in wellnames_2]+ [w+"_changed" for w in wellnames_2]
robust_res_columns_SOS2 = base_res+ [w+"_changed" for w in wellnames_2]
robust_res_columns_recourse = base_res[1:]+[w+"_gas_var" for w in wellnames_2]+ [w+"_changed" for w in wellnames_2]

robust_eval_columns = ["inf_tot", "inf_indiv", "tot_oil", "tot_gas"]+[w+"_gas_mean" for w in wellnames_2]+[w+"_oil_mean" for w in wellnames_2]+[w+"_oil_var" for w in wellnames_2]+[w+"_gas_var" for w in wellnames_2]

recourse_algo_columns = ["infeasible count", "oil output", "gas output"]+ [w+"_choke_final" for w in wellnames_2]
indiv_cap = 54166
tot_cap = 250000

phasenames = ["oil", "gas"]
param_dict = {'dropout':[x for x in np.arange(0.05,0.4,0.1)], 'regu':[1e-6, 1e-5, 1e-4, 1e-3, 1e-2], 'layers':[1,2], 'neurons':[20,40]}
param_dict_rand = {'dropout':uniform(0.01, 0.4),
                  'regu':uniform(1e-6, 1e-4),
                  'layers':randint(1,3),
                  'neurons':randint(5,40)}

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
                lower[well][sep] = max(0.0, 0.9*dfw.min()-0.01) #do not allow negative choke values
                upper[well][sep] = min(100.0, 1.3*dfw.max()+0.01)
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

def load_2(well, phase, separator="HP", case=1, mode = "mean", scen=0):
    if mode == "scen":
        filename = "scenarios/nn/points/"+well+"_"+str(scen)+"-scen-"+phase+".txt"
    else:
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
                   case=1, mode="mean", folder = "weights\\", num=""):
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
    filename = folder + datafile + "-" + sep + "-" + goal + num
    print("Filename:", filename)
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

def get_factors(init_name):
    cols = [w+"_factor" for w in wellnames_2]
    df = pd.read_csv("results/initial/res_initial.csv", sep=";")
    df = df.loc[df["name"]==init_name]
    df = df[cols]
    df.columns=wellnames_2
    df = df.loc[df.index[0]]
    return df

def generate_scenario_trunc_normal(init_name, num_scen, sep="HP", phase="gas", lower=-4, upper=4, folder="", iteration=None, distr="truncnorm", predeterm=None):
    if(distr=="truncnorm"):
        scen = ss.truncnorm.rvs(lower, upper, size=(num_scen,len(wellnames_2)))
    elif(distr=="triang"):
        scen = np.random.triangular(lower, 0, upper, size=(num_scen,len(wellnames_2)))
    elif(distr=="eev"):
        num_scen=1
        scen = np.zeros((num_scen, len(wellnames_2)))
    else:
        raise ValueError("Unknown distribution specified.")
        
    
#    df = pd.DataFrame(pd.Series(scen), dtype=wellnames_2)
    df = pd.DataFrame(columns=wellnames_2)
    for i in range(num_scen):
        df.loc[i] = scen[i]

    #get scenario factors and apply
    factors = get_factors(init_name)
    for w in wellnames_2:
        if(factors[w] != math.inf):
            df[w] = factors[w]
    filename = "scenarios/"+init_name+"_"+phase+"_"+str(num_scen)+"_"+str(lower)+"_"+str(upper)+((" ("+str(iteration)+")") if iteration else "")+"_"+distr+".csv"
    with open(filename, 'w') as f:
        df.to_csv(f,sep=";", index=False)
        
        
def load_scenario(init_name, num_scen, lower, upper, phase, sep, iteration=None, distr="truncnorm"):
    filename = "scenarios/"+init_name+"_"+phase+"_"+str(num_scen)+"_"+str(lower)+"_"+str(upper)+((" ("+str(iteration)+")") if iteration else "")+"_"+distr+".csv"
    df = pd.read_csv(filename, sep=';')
    return df

def get_scenario(init_name, num_scen, lower=-4, upper=4, phase="gas", sep="HP", iteration=None, distr="truncnorm"):
    #iteration flag determines if we wish to create a new version of the specified scenario number
    #use the flag for calc during in-sample/out-of-sample stability testing
    try:
        return load_scenario(init_name, num_scen, lower, upper, phase, sep, iteration=iteration, distr=distr)
    except Exception as e:
        generate_scenario_trunc_normal(init_name, num_scen, sep=sep, phase=phase, lower=lower, upper=upper, iteration=iteration, distr=distr)
        return load_scenario(init_name, num_scen, lower, upper, phase, sep, iteration=iteration, distr=distr)
    
def get_robust_solution(num_scen=100, lower=-4, upper=4, phase="gas", sep="HP", init_name=None):
    if(init_name):
        df = pd.read_csv("results/initial/res_initial.csv", sep=";")
        df = df.loc[df["name"]==init_name]
        df.drop(["name"], axis=1, inplace=True)
#        df = pd.DataFrame(np.concatenate((df.values,np.zeros((1,7))), axis=1), columns=robust_res_columns)
#        df2 = pd.DataFrame(np.zeros((1,14)), columns=[w+"_gas_var" for w in wellnames_2]+[w+"_changed" for w in wellnames_2])
#        df = pd.concat([df, df2], axis=1)
        return df
    else:
        df = pd.read_csv("results/robust/res.csv", sep=";")
        df = df.loc[df["scenarios"]==num_scen]
    if(df.shape[0] ==0 ):
        raise ValueError("Specified scenario number or initial name not in file!")
    c = [w+"_choke" for w in wellnames_2]
    indiv_cap = df["indiv_cap"].values[0]
    tot_cap = df["tot_cap"].values[0]
    for w in wellnames_2:
        if df[w+"_oil_mean"].values[0] == 0:
            df[w+"_choke"] = 0
            
    df_ret = df[c]
    df_ret.columns = wellnames_2
    return df_ret, indiv_cap, tot_cap

def extract_xy(df_w):
    xcols = [w+"_choke" for w in wellnames_2]
    ycols = [w+"_gas_mean" for w in wellnames_2]
    x_ = df_w[xcols]
    y_ = df_w[ycols]
    x_.columns = wellnames_2
    y_.columns = wellnames_2
    x_= x_.loc[x_.index[0]]
    y_= y_.loc[y_.index[0]]

    return x_.to_dict(), y_.to_dict()


def get_init_chokes(init_name):
    df = pd.read_csv("results/initial/res_initial.csv", sep=";")
    df = df.loc[df["name"]==init_name]
    df.drop(["name"], axis=1, inplace=True)
    cols = [w+"_choke" for w in wellnames_2]
    df = df[cols]
    return {w:df[w+"_choke"].values[0] for w in wellnames_2}

# =============================================================================
# build a ReLU NN from dims, weights and bias
# =============================================================================
def retrieve_model(well, goal="oil", lr=0.001, case=2, mode="mean"):
    dims, w, b = load_2(well,goal,case=case, mode=mode)
    model_1= Sequential()
    for i in range(1,len(dims)):
        new_w = [np.array(w[i-1]), np.array(b[i-1])]
        model_1.add(Dense(dims[i], input_shape=(dims[i-1],),
                          weights = new_w))
        if (i == len(dims)-1):
            model_1.add(Activation("linear"))
        else:
            model_1.add(Activation("relu"))
    model_1.compile(optimizer=optimizers.Adam(lr=lr), loss="mse")
    return model_1

#def retrieve_model(dims, w, b, lr=0.001):
#    model_1= Sequential()
#    for i in range(1,len(dims)):
#        new_w = [np.array(w[i-1]), np.array(b[i-1])]
#        model_1.add(Dense(dims[i], input_shape=(dims[i-1],),
#                          weights = new_w))
#        if (i == len(dims)-1):
#            model_1.add(Activation("linear"))
#        else:
#            model_1.add(Activation("relu"))
#            model_1.add(Dropout(0.05))
#    model_1.compile(optimizer=optimizers.Adam(lr=lr), loss=sced_loss)
#    return model_1

def build_and_plot_well(well, goal="oil", case=2):
    model = retrieve_model(well,goal=goal,lr=0.001,case=case)
    X = np.array([[i] for i in range(101)])
    pred = [x[0] for x in model.predict(X)]
    print(pred)
    return pred
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    line2 = ax.plot(X, pred, color='green',linestyle='dashed', linewidth=1)
    pyplot.xlabel('choke')
    pyplot.ylabel("oil")
    pyplot.show()
    

#def save_variance_func_2(X, var, mean, case, well, phase):
#    filename = "variance_case" + str(case) +"_"+phase+".csv"
#    old = pd.read_csv(filename,sep=";",index_col=0)
#    d = {well+"_"+phase+"_mean": old[well+"_"+phase+"_mean"], well+"_"+phase+"_var": var ,well+"_"+phase+"_X":X}
#    try:
#        df = pd.read_csv(filename+"_", sep=';', index_col=0)
#        for k, v in d.items():
#            df[k] = v
#    except Exception as e:
#        print(e)
#        df = pd.DataFrame(data=d)
#        print(df.columns)
#    with open(filename+"_", 'w') as f:
#        df.to_csv(f,sep=";")

def save_variance_func(X, var, mean, case, well, phase):
    filename = "variance_case" + str(case) +"_"+phase+".csv"
#    well+'_'+phase+"_std":var, 
    try:
        df = pd.read_csv(filename, sep=';', index_col=0)
#        old = pd.read_csv("variance_case_2.csv",sep=";",index_col=0)
        d = {well+"_"+phase+"_mean": mean, well+"_"+phase+"_var": var, well+"_"+phase+"_X":X}
        for k, v in d.items():
            df[k] = v
    except Exception as e:
        print(e)
        df = pd.DataFrame(data=d)
        print(df.columns)
    with open(filename, 'w') as f:
        df.to_csv(f,sep=";")
        
    
        
def sample_mean_std(model, X, n_iter, f):
    #gather results from forward pass
    results = np.column_stack(f((X,1.))[0])
    if (np.isnan(results[0][0])):
        print("NAN")
        return

    res_mean = [results[0]]
    res_var = [np.exp(results[1])]
    for i in range(1,n_iter):
        a = np.column_stack(f((X,1.))[0])
        res_mean.append(a[0])
        res_var.append(np.exp(a[1]))
        
    pred_mean = np.mean(res_mean, axis=0)
    pred_sq_mean = np.mean(np.square(res_mean), axis=0)
    var_mean = np.mean(res_var, axis=0)
    std = np.sqrt(pred_sq_mean-np.square(pred_mean)+var_mean)
    return pred_mean, std

def sample_mean_2var(model, X, n_iter, f):
    #gather results from forward pass
    results = np.column_stack(f((X,1.))[0])
    if (np.isnan(results[0][0])):
        print("NAN")
        return

    res_mean = [results[0]]
    res_var = [np.exp(results[1])]
    for i in range(1,n_iter):
        a = np.column_stack(f((X,1.))[0])
        res_mean.append(a[0])
        res_var.append(np.exp(a[1]))
        
    pred_mean = np.mean(res_mean, axis=0)
    pred_sq_mean = np.mean(np.square(res_mean), axis=0)
    var_mean = np.mean(res_var, axis=0)
#    std = np.sqrt(pred_sq_mean-np.square(pred_mean)+var_mean)
    epi = pred_sq_mean-np.square(pred_mean)
    return pred_mean, var_mean, epi

def mean_var_to_csv(well, phase="gas", mode="mean", n_iter=200, case=2):
    dims, w, b = load_2(well, phase=phase, case=2, mode=mode)
#    print(w)
#    print(b)
    model = retrieve_model(dims,w,b)
#    for i in range(0,7,3):
#        print(model.layers[i].get_weights())
    X_test = np.array([[x] for x in range(101)])
    f = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
    mean, std = sample_mean_std(model, X, n_iter, f)
    var = std**2
#    pred_mean, std = sample_mean_std(model, X_test, n_iter,
#                                     K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output]))
#    plot_once(X_test, 2, pred_mean, std, 1, 1)
    save_variance_func([x for x in range(101)], None, pred_mean, case, well, phase)
    
def inverse_scale(model_1, dim, neurons, dropout, rs, lr, loss):
    model_2= Sequential()
    model_2.add(Dense(neurons, input_shape=(dim,), weights = [model_1.layers[0].get_weights()[0].reshape(dim,neurons),
                      rs.inverse_transform(model_1.layers[0].get_weights()[1].reshape(-1,1)).reshape(neurons,)]))
    model_2.add(Activation("relu"))
    model_2.add(Dropout(dropout))
    model_2.add(Dense(neurons, weights = [model_1.layers[3].get_weights()[0].reshape(neurons,neurons),
                      rs.inverse_transform(model_1.layers[3].get_weights()[1].reshape(-1,1)).reshape(neurons,)]))
    model_2.add(Activation("relu"))
    model_2.add(Dropout(dropout))
    model_2.add(Dense(2,  weights = [model_1.layers[-2].get_weights()[0],rs.inverse_transform(model_1.layers[-2].get_weights()[1].reshape(-1,1)).reshape(2,)]))
    model_2.add(Activation("linear"))
    model_2.compile(optimizer=optimizers.adam(lr=lr), loss = loss)
    return model_2

def add_layer(model_1, neurons, loss, factor=1000000.0):
    model_2= Sequential()
    model_2.add(Dense(neurons, input_shape=(1,), weights = [model_1.layers[0].get_weights()[0].reshape(1,neurons),
                      model_1.layers[0].get_weights()[1].reshape(-1,1).reshape(neurons,)]))
    model_2.add(Activation("relu"))
    model_2.add(Dense(1, weights = [model_1.layers[2].get_weights()[0].reshape(neurons,1),
                      model_1.layers[2].get_weights()[1].reshape(-1,1).reshape(1,)]))
    model_2.add(Activation("linear"))
    model_2.add(Dense(1, weights = [np.array([[factor]]), np.array([0.0])], trainable=False))
    model_2.compile(optimizer=optimizers.adam(lr=0.001), loss = loss)
    return model_2

def get_sos2_scenarios(phase, num_scen, init_name="", iteration=None):
    dbs = {}
    chks = {}
    folder = "scenarios\\nn\\points\\"+("stability\\" if iteration is not None and phase=="gas" else "")
    if(num_scen=="eev"):
        df = pd.read_csv(folder+"sos2_"+phase+"_"+init_name+((" ("+str(iteration)+")") if iteration and phase=="gas" else "")+("_eev" if phase=="gas" else "")+".csv", delimiter=";", header=0)
        num_scen=1
#        dbs[0] = {}
#        for well in wellnames_2:
#            dbs[0][well] = df[well+"_"+phase+"_"+str(0)]
#            chks[0][well] = df[well+"_choke"]
#        return dbs, chks
    else:
        df = pd.read_csv(folder+"sos2_"+phase+"_"+init_name+((" ("+str(iteration)+")") if iteration and phase=="gas" else "")+".csv", delimiter=";", header=0)
    avail_scenarios=(len(df.keys())-2)/7
    if phase=="gas":
        for i in range(int(num_scen)):
            if (i>=avail_scenarios):
                print("Not enough generated scenarios")
                break
            dbw ={}
            for well in wellnames_2:
                dbw[well] = df[well+"_"+phase+"_"+str(i)]
            dbs[i]=dbw
    elif phase=="oil":
        for well in wellnames_2:
            dbs[well] = df[well+"_"+phase+"_"+str(0)]
    #get choke vals
    for well in wellnames_2:
        chks[well] = df[well+"_choke"]
    return dbs, chks

#TODO: modify to load true
def get_sos2_true_curves(phase, init_name, iteration):
    dbs = {}
    chks = {}
    if phase=="oil":
        iteration=0
        true_string = ""
    else:
        true_string = "_true"
    df = pd.read_csv("scenarios\\nn\\points\\sos2_"+phase+"_"+init_name+true_string+".csv", delimiter=";", header=0)
    if iteration is None:
        addstr = ""
    else:
        addstr = "_"+str(iteration)
    for well in wellnames_2:
        dbs[well] = df[well+"_"+phase+addstr]
    for well in wellnames_2:
        chks[well] = df[well+"_choke"]
    return dbs, chks

