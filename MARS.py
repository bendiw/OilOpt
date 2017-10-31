import numpy as np
from pyearth import Earth
import pandas as pd
from matplotlib import pyplot
import math
import caseloader as cl


base_dir = "C:\\Users\\Bendik\\Documents\\GitHub\\OilOpt\\"
data_file = "welltests.csv"

df = pd.read_csv(base_dir+data_file, sep=",")
df = df[pd.notnull(df['gaslift_rate'])]
df = df[pd.notnull(df['oil'])]

well='A5'
model = Earth(allow_missing = True, enable_pruning=False, max_terms=10, penalty=0,
              minspan=3)
wells = df.well.unique()
for well in ['A5']:
    df_w = df.loc[df['well'] == well]
    df = cl.load(base_dir+data_file)
    X,y = cl.gen_targets(df, well, normalize=True, intervals=20)
    X = np.array(X)
    y = np.array(y)
    print("\nwell: ", well)
    model.fit(X, y)

##    print(model.trace())
##    print(model.summary())

    y_hat = model.predict(X)
    pyplot.figure()
    pyplot.plot(X,y,'r.')
    pyplot.plot(X,y_hat,'b.')
    pyplot.xlabel('gaslift')
    pyplot.ylabel('oil')
    pyplot.title(well)
    pyplot.show()
