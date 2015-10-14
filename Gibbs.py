import numpy as np
import distrib

# need to clean up this function

def run_gibbs(data, reps, sites, k):
    #loop through the gibbs sampler from one initial value for the components
    #data parameters
    n = data.shape[1]

    #prior paramters for Normal Gamma
    alpha_0 = 1   
    beta_0 = .5
    mu_0 = 6
    kappa_0 = 1
    #prior parameters for Dirichlet
    prior_cnt = np.ones(k, dtype = np.int8)*2
    alpha = np.empty(k)

    #Gibbs Sampler parameters
    post_mu = np.empty([reps, sites , k])
    post_prec = np.empty([reps, sites, k])
    lik = np.empty((k, n))
    logP_h0 = np.empty(sites)
    comp = np.empty((reps+1, sites, n), dtype =  np.int8)

    #initialize Gibbs sampler by randomly assigning component to each observation
    comp[0] = np.random.randint(k, size = (sites, n))  

    for r in range(reps):
        alpha = distrib.updateDirichlet(comp[r], prior_cnt)
        for s in range(sites):
    #sample mean and precision from a Normal Gamma
            post_mu[r, s, :], post_prec[r, s, :] = distrib.NG_draws(data[s], comp[r,s], k, kappa_0, mu_0, alpha_0, beta_0)
    #sort mean from least to greatest
            post_mu[r, s, :], post_prec[r, s, :] = distrib.sort(post_mu[r, s, :], post_prec[r, s,:])
    #determine likelihood of each observation being sampled from each component normal distribution
            lik = distrib.calc_lik(data[s], post_mu[r, s,:], post_prec[r, s,:])
    #sample multinoulli parameters for each site
            pi = np.random.dirichlet(alpha[s])    
    #calculate probability of belonging to each component for each data point
            comp_prob = pi[:, np.newaxis]*lik
    #normalize comp_prob
            comp_prob = comp_prob / np.sum(comp_prob, axis = 0)
    #sample for z using the derived posterior probability
            comp[r+1, s] = distrib.vec_multin(comp_prob)
    #       logP_h0[s] = marg_dirich(comp[j+1])
    
    return post_mu, post_prec, comp