from __future__ import division
from scipy.stats import norm
import numpy as np
import simulate_data
import distributions as distrib

sites = 30
reps = 1500
k = 2
sampling_rounds = 4

data = simulate_data.simulate(sites, k, np.array([10]))

#initialize the distributions
z = distrib.Components(data, reps)
theta = distrib.NG(data, reps)
weights = distrib.MultiDirich(data)

for r in range(reps):
#draw the kernel components
    z.update(data, theta, weights.Pi)
#use likelihood of values and dirichlet prior to draw mixing weights
    weights.update(data, z.kernel_comp[z.counter])
#sample mean and precision from a Normal Gamma
    theta.update(data, z.kernel_comp[z.counter])