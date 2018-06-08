import pandas as pd
import numpy as np
import math
import tools as t
from sklearn.preprocessing import normalize, RobustScaler, StandardScaler


def load(path, well=None, case=1):
    if case!=1:
        index_col = 0
    else:
        index_col = None
    df = pd.read_csv(path,sep=',', index_col=index_col)
    if(well and case==1):
        df = df.loc[df['well'] == well]
    return df

def gen_test_case(cases=30):
    X, Y = [], []
    data = []
    for i in range(cases):
        x = r.uniform(-10,10)
        X.append(x)
    X.sort()
    for x in X[:cases-5]:
        y = x**2 + r.uniform(-abs(x),abs(x))
        Y.append(y)
        data.append([[x],[y]])
    for x in X[cases-5:]:
        y=x/2
        Y.append(y)
        data.append([[x],[y]])
    return data


def BO_load(well, separator="HP",case=1,  goal="oil", scaler="rs", nan_ratio = 0.3):
    if separator == "HP":
        hp=1
    else:
        hp=0
# =============================================================================
#     load and normalize data
# =============================================================================
    if case==1:
        df = load("welltests_new.csv")
        dict_data,_,_ = gen_targets(df, well+"", goal=goal, normalize=False, intervals = 20, factor = 1.5, nan_ratio = nan_ratio, hp=hp) #,intervals=100
        data = t.convert_from_dict_to_tflists(dict_data)
    else:
        goal = goal.upper()
        print("goal:", goal)
        df = load("dataset_case2\case2-oil-dataset.csv", case=2).dropna(subset=[well+'_Q'+goal+'_wsp_mea', well+'_CHK_mea'])
        X = df.as_matrix(columns=[well+'_CHK_mea'])
        y = df.as_matrix(columns=[well+'_Q'+goal+'_wsp_mea'])
        data = np.array([[X[i], y[i]] for i in range(len(X))])

    if (len(data[0][0]) >= 2):
        is_3d = True
        dim = 2
    else:
        is_3d = False
        dim=1
    if scaler=="rs":
        print("Robust scaling of data")
        rs = RobustScaler(with_centering =False)
    elif scaler == "std":
        rs = StandardScaler(with_mean=False)
        print("Standard scaling of data")
    else:
        scaler=None
        rs=None

    if is_3d:
        glift_orig = np.array([x[0][0] for x in data])
        choke_orig = np.array([x[0][1] for x in data])
        y_orig = np.array([x[1][0] for x in data]).reshape(-1,1)
        glift = rs.fit_transform(glift_orig.reshape(-1,1))
        choke = rs.transform(choke_orig.reshape(-1,1))
        y = rs.transform(y_orig.reshape(-2, 1))
        X = np.array([[glift[i][0], choke[i][0]] for i in range(len(glift))])
    else:
        X = np.array([x[0][0] for x in data]).reshape(-1,1)
        y = np.array([x[1][0] for x in data]).reshape(-1,1)
#        scaler = StandardScaler().fit(X, y)
        if(scaler):
            if goal.lower()=="gas":
                y=y/1000
            y = rs.fit_transform(y.reshape(-1,1))
            X = rs.transform(X.reshape(-1,1))
#            X = rs.fit_transform(X.reshape(-1,1))
#            y = rs.transform(y.reshape(-1, 1))

    return X, y, rs

##############
##              Generates targets from a dataframe                      ##
## If intervals input is given, the dataset will be processed to reduce ##
## amount of data points in close proximity to each other. The way this ##
## is done is specified by the 'mode' parameter.                        ##
##############
def gen_targets(df, well, goal='oil', intervals=None, allow_nan=False, normalize = False, factor=0, nan_ratio=0.5, hp=True):
    df = df.loc[df['well']==well]
    if(df['prs_dns'].isnull().sum()/df.shape[0] >= 0.7):
        df = df
    elif(hp==1):
        df = df.loc[df['prs_dns']>=18.5]
    else:
        df = df.loc[df['prs_dns']<18.5 ]

    ret = {}
    add_Z = False
    if not (df['choke'].isnull().sum()/df.shape[0] > nan_ratio):
        add_Z = True
        intervals = None
        Z = []
    else:
        Z = None
    if(not allow_nan):
        df = df[pd.notnull(df['gaslift_rate'])]
        df = df[pd.notnull(df[goal])]
        if(add_Z):
            df = df[pd.notnull(df['choke'])]
    if(intervals):
        X = []
        y = []
        min_g = df['gaslift_rate'].min()
        max_g = df['gaslift_rate'].max()
        g_diff = max_g-min_g
        step = g_diff/intervals
        vals = np.arange(min_g, max_g+step, step)
        for i in range(len(vals)):
            val = df.loc[df['gaslift_rate']>=vals[i]]
            val = val.loc[val['gaslift_rate']<=vals[i]+step]
            maxtime = val['time_ms_begin'].max()
            mintime = val['time_ms_begin'].min()
            div = maxtime-mintime
            if(val.shape[0] > 1):
                factors = val['time_ms_begin'].apply(lambda x: math.exp(-factor*((maxtime-x)/div)))
                oil = val[goal].multiply(factors).sum() / factors.sum()
                glift = val['gaslift_rate'].multiply(factors).sum() / factors.sum()
            elif(val.shape[0]==1):
                glift = val['gaslift_rate'].values[0]
                oil = val[goal].values[0]
            if(not val.empty):
                X.append(glift)
                y.append(oil)
    else:
        if add_Z:
            Z = df['choke']
        X = df['gaslift_rate']
        y = df[goal]
        
    if(normalize):
        X, X_mean, X_std = normalize_data(X)
        y, y_mean, y_std = normalize_data(y)
        means = [X_mean]
        stds = [X_std]
        if add_Z:
            Z, Z_mean, Z_std = normalize_data(Z)
            means.append(Z_mean)
            stds.append(Z_std)
        means.append(y_mean)
        stds.append(y_std)
    else:
        means=None
        stds = None
    ret['gaslift'] = np.array(X)
    ret['output'] = np.array(y)
    if(add_Z):
        ret['choke'] = np.array(Z)
    return ret, means, stds

def normalize_data(data):
    X_mean = np.mean(data)
    X_std = np.std(data)
    X = [(x-X_mean)/X_std for x in data]
    return X, X_mean, X_std

def conv_to_batch_multi(X, Y, Z):
    batch = []
    for i in range(len(X)):
        tup = [ [X[i], Y[i]], Z[i] ]
        batch.append(tup)
    return batch

def conv_to_batch(data):
    batch= []
    for i in range(len(data[0])):
        tup = [[data[0][i]], [data[1][i]]]
        batch.append(tup)
    return batch
