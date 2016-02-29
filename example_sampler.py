from __future__ import division
from scipy.stats import norm
import numpy as np
import fetch_data
import distrib

sites = 3
reps = 10
k = 3

data = fetch_data.simulate(sites, k, np.array([60]))
#loop through the gibbs sampler to set initial value for the parameter
z = distrib.Components(data, 10)
theta = distrib.NG(data, 10)
weights = distrib.MultiDirich(data)

for r in range(reps):
#draw components
    z.update(data, theta, weights.Pi)
#use likelihood of values and dirichlet prior to draw mixing weights
    weights.update(data, z.kernel_comp[z.counter])
#sample mean and precision from a Normal Gamma
    theta.update(data, z.kernel_comp[z.counter])