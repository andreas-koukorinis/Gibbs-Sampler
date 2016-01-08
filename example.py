from __future__ import division
from scipy.stats import norm
import numpy as np
import set_params
import distrib as dist

sites = 3
reps = 3000
k = 3

data = set_params.simulate(sites, k, np.array([30]))
#loop through the gibbs sampler to set initial value for the parameter
z = dist.Components(reps, data)
theta = dist.NG(reps, data)
phi = dist.MultiDirich(sites, data)

for r in range(reps):
#draw components
    z.update_comp(data.vals, theta, phi.pi, r)
#sample mean and precision from a Normal Gamma
    theta.update(data, z.comp[r], r)
#use likelihood of values and dirichlet prior to draw mixing weights
    phi.draw_dirichlet(z.comp[r])