# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:46:54 2017

@author: Bendik
"""

import pandas as pd
from matplotlib import pyplot
wellnames = ["A2", "A3", "A5", "A6", "A7", "A8", "B1", "B2", 
             "B3", "B4", "B5", "B6", "B7", "C1", "C2", "C3", "C4"]
wellnames2 = ["W"+str(i) for i in range(1,8)]
well_to_sep = {"A2" : ["HP"], "A3": ["HP"], "A5": ["HP"], "A6": ["HP"], "A7": ["HP"], "A8": ["HP"], 
               "B1" : ["HP", "LP"], "B2" : ["HP", "LP"], "B3" : ["HP", "LP"], "B4" : ["HP", "LP"], "B5" : ["HP", "LP"], "B6" : ["HP", "LP"], "B7" : ["HP", "LP"], 
               "C1" : ["LP"], "C2" : ["LP"], "C3" : ["LP"], "C4" : ["LP"]}
#df_new = pd.read_csv("C:\\Users\\Bendik\\Documents\\GitHub\\OilOpt\\welltests_csv.csv", delimiter=",", header=0)
#df_old = pd.read_csv("C:\\Users\\Bendik\\Documents\\GitHub\\OilOpt\\welltests_old.csv", delimiter=",", header=0)
df = pd.read_csv("dataset_case2\case2-oil-dataset.csv", delimiter=",", header=0, index_col=0)
print(df.columns)

for w in wellnames2:
    x = df[w+'_CHK_mea']
    y = df[w+'_QGAS_wsp_mea']
    fig = pyplot.figure()
    pyplot.title(w)
    ax = fig.add_subplot(111)
    ax.plot(x, y, linestyle='None', marker = '.',markersize=15)
    pyplot.show()
#for well in wellnames:
#    for sep in well_to_sep[well]:
#        dfw = df.loc[df["well"]==well]
#        print(well, sep)
#        if(dfw["prs_dns"].isnull().sum()/dfw.shape[0] >= 0.7):
#            print(dfw["gaslift_rate"].min(), dfw["gaslift_rate"].max())
#        elif(sep=="HP"):
#            print(dfw.loc[dfw["prs_dns"]>= 18.5]["gaslift_rate"].min(), dfw.loc[dfw["prs_dns"]>= 18.5]["gaslift_rate"].max())
#        else:
#            print(dfw.loc[dfw["prs_dns"]< 18.5]["gaslift_rate"].min(), dfw.loc[dfw["prs_dns"]< 18.5]["gaslift_rate"].max())
#        print("\n")
#print(df.loc[df["well"]=="Well 0"])
#print(df_new.loc[df_new["well"]=="Well 31"].shape)
#print(df_new.shape)
#print(df_old.shape)

#drops = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'A3', 'A8', 'C1']
#new = ['Well 72','Well 151','Well 95','Well 34','Well 9','Well 179', 'Well 120', 'Well 193', 'Well 136']

#df_old = df_old.loc[df_old["well"] ]

#for d in drops:
#    s = df_old.shape[0]
#    df_old = df_old.loc[df_old["well"]!=d]
#    print(d, "dropped", s-df_old.shape[0], "rows.")
#    
#
#print(df_old.shape)
#
#for n in range(len(new)):
#    to_add = df_new.loc[df_new["well"]==new[n]]
#    to_add.loc[:, "well"] = drops[n]
##    print(to_add["well"])
#    df_old = df_old.append(to_add)
#    print(new[n], "added", to_add.shape[0], "rows.")
#print(df_old.shape)
#print(df_old.loc[(df_old["well"]=="B5") & (df_old["gaslift_rate"]==0), ["gaslift_rate", "choke"]])
##df_old = df_old[~(df_old["well"]=="B2") & (df_old["gaslift_rate"]==0)]
#print(df_2.loc[df_2["well"]=="B5"]["gaslift_rate"])
#print(df_old.shape)
#print(df_old.loc[df_old["well"]=="B2" & ~df_old["gaslift_rate"]==0])
#
#df_old.to_csv("C:\\Users\\Bendik\\Documents\\GitHub\\OilOpt\\welltests_new.csv", sep=",", index=False)
#    
#  
#data=[]
#for well in df_new.well.unique():
#    wr = []
#    dfw = df_new.loc[df_new["well"]==well]
#    wr.append(dfw.shape[0])
#    wr.append(dfw["prs_dns"].isnull().sum())
#    wr.append([well])
#    wr.append()
#    wr.append(dfw.loc[dfw["prs_dns"]< 18.5].shape[0])
#    wr.append(dfw.shape[0]-dfw["gaslift_rate"].isnull().sum())
#    wr.append(dfw.shape[0]-dfw["choke"].isnull().sum())
#    wr.append(dfw.shape[0]-dfw["gas"].isnull().sum())
#    wr.append(dfw.shape[0]-dfw["oil"].isnull().sum())
#
#
#    data.append(wr)
##    dfw.loc[dfw["prs_dns"]< 18.5].shape[0])
#
##    print(df)
#data = sorted(data, key=lambda x: x[0])
#for r in data[30:130]:
#    print(r[0])
#    print("*********** WELL", r[2], "***********")
#    print("data rows:",r[0]) 
#    print("missing pressure:",r[1] )
#    print("HP sep:", r[3], "\t %:", r[3]/r[0])
#    print("LP sep:", r[4], "\t %:", r[4]/r[0])
#    print("gaslift:", r[5], "\t %:", r[5]/r[0])
#    print("choke:", r[6], "\t %:", r[6]/r[0])
#    print("gas", r[7], "\t\t %:", r[7]/r[0])
#    print("oil", r[8], "\t\t %:", r[8]/r[0])
#
#    print("\n\n")
#    
#print("total # of wells:", len(df_new.well.unique()))
#      
#print(df_new.loc[df_new["well"]== "Well 151"][["gaslift_rate", "choke"]])

#print(df_old.loc[(df_old["well"]=="B2") ]["gaslift_rate"])
#z = df_2.loc[df_2["well"]=="B2"]
#print(z.loc[z["prs_dns"] < 18.5])