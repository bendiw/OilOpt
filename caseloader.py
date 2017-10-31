import pandas as pd

def load(path, well=None):
    df = pd.read_csv(base_dir+data_file, sep=",")
    if(well):
        df = df.loc[df['well'] == well]
    return df

def gen_targets(df, well=None, intervals=100):
    step = g_diff/intervals
    X = []
    y = []
    vals = np.arange(min_g, max_g, step)
    #print(df_w.loc[df_w['gaslift_rate']>0])
    for i in vals:
        val = df_w.loc[df_w['gaslift_rate']>=i-step]
        val = val.loc[val['gaslift_rate']<=i+step]
        if(val.shape[0] >= 1):
            glift, oil = val.ix[val['time_ms_begin'].idxmax()][['gaslift_rate', 'oil']]
        elif(not val.empty):
            glift, oil = val[['gaslift_rate','oil']]
        if(not val.empty):
            X.append(glift)
            y.append(oil)
