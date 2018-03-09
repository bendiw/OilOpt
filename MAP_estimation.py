# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 12:10:40 2018

@author: bendi
"""

import pymc3 as pm
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.preprocessing import normalize, RobustScaler
import caseloader as cl
import tens
import matplotlib.pyplot as plt

# set the seed
np.random.seed(1)

n = 100 # The number of data points
X = np.linspace(0, 10, n)[:, None] # The inputs to the GP, they must be arranged as a column vector


def run(well="A5", sep="HP", hp=1, goal="oil", intervals = 20, factor = 1.5, nan_ratio = 0.3):
    data = load_well(well, sep, goal, hp, factor, intervals, nan_ratio)
    rs = RobustScaler(with_centering =False)
    X_orig = np.array([x[0][0] for x in data]).reshape(-1,1)
    y_orig = np.array([x[1][0] for x in data]).reshape(-1,1)
    X = rs.fit_transform(X_orig.reshape(-1,1))
    n = X.shape[0]
    y = rs.transform(y_orig.reshape(-1, 1)).reshape(n,)
    σ_true = 2.0
    # Define the true covariance function and its parameters
    ℓ_true = 1.0
    η_true = 3.0
    mean_func = pm.gp.mean.Zero()
#    n = 100
    cov_func = η_true**2 * pm.gp.cov.Matern52(1, ℓ_true)
    
#    X = np.linspace(0, 10, 100)[:, None] # The inputs to the GP, they must be arranged as a column vector
#    f_true = np.random.multivariate_normal(mean_func(X).eval(),
#                                       cov_func(X).eval() + 1e-8*np.eye(n), 1).flatten()
#    y = f_true + σ_true * np.random.randn(n)
    # The latent function values are one sample from a multivariate normal
    # Note that we have to call `eval()` because PyMC3 built on top of Theano
    
    # The observed data is the latent function plus a small amount of IID Gaussian noise
    # The standard deviation of the noise is `sigma`
    #y = f_true + σ_true * np.random.randn(n)
    with pm.Model() as model:
#        ℓ = pm.Gamma("ℓ", alpha=2, beta=1)
        ℓ = pm.Normal("ℓ", mu=1, sd=0.5)
        η = pm.HalfCauchy("η", beta=5)
    
#        cov = η**2 * pm.gp.cov.Matern52(1, ℓ)
        cov = η**2 * pm.gp.cov.Exponential(1, ℓ)
        gp = pm.gp.Marginal(cov_func=cov)
    
        σ = pm.HalfCauchy("σ", beta=5)
        y_ = gp.marginal_likelihood("y", X=X, y=y, noise=σ)
        mp = pm.find_MAP()
    
    df = pd.DataFrame({"Parameter": ["ℓ", "η", "σ"],
              "Value at MAP": [float(mp["ℓ"]), float(mp["η"]), float(mp["σ"])],
              "True value": [ℓ_true, η_true, σ_true]})
    print(df)
    
#    X_new = np.linspace(0, 20, 600)[:,None]
    # add the GP conditional to the model, given the new X values
    with model:
        f_pred = gp.conditional("f_pred", X)
    
    # To use the MAP values, you can just replace the trace with a length-1 list with `mp`
    with model:
        pred_samples = pm.sample_ppc([mp], vars=[f_pred], samples=100)
    
    
    # plot the results
    fig = plt.figure(figsize=(12,5)); ax = fig.gca()
    
    # plot the samples from the gp posterior with samples and shading
    from pymc3.gp.util import plot_gp_dist
    plot_gp_dist(ax, pred_samples["f_pred"], X);
    
    # plot the data and the true latent function
#    plt.plot(X, f_true, "dodgerblue", lw=3, label="True f");
    plt.plot(X, y, 'ok', ms=3, alpha=0.5, label="Observed data");
    
    # axis labels and title
    plt.xlabel("X"); plt.ylim([-max(y)*0.5,max(y)*1.2]);
    plt.title("Posterior distribution over $f(x)$ at the observed values")
    plt.show()
    
    
def load_well(well, separator, goal, hp, factor, intervals, nan_ratio):
    df = cl.load("welltests_new.csv")
    dict_data,_,_ = cl.gen_targets(df, well+"", goal=goal, normalize=False, intervals=intervals,
                               factor = factor, nan_ratio = nan_ratio, hp=hp) #,intervals=100
    data = tens.convert_from_dict_to_tflists(dict_data)
    return data