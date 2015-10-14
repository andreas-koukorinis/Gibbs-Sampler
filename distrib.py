from __future__ import division
import numpy as np
from scipy.stats import norm


def vec_multin(p):
    #draw samples from n multinoulli distributions
    #p is a matrix where each column sums to one
    #number of columns equals number of samples
    n = p.shape[1]
    pcum = p.cumsum(axis = 0)
    rvs = np.random.rand(1,n)
    return (rvs<pcum).argmax(0)

def NG_draws(data, comp, k, kappa_0, mu_0, alpha_0, beta_0):
    x = np.empty(k)
    prec = np.empty(k)
    for i in range(k):
#retrieve observations from component k
        data_k = data[comp == i]
        n = data_k.size
#determine sample statistics of observations in component k
        data_var = np.var(data_k)
        data_avg = np.mean(data_k)
        if n == 0:
            data_var = 0
            data_avg = 0
#sample for Normal distribution parameters
#set parameters for Normal Gamma
        mu_1 = (kappa_0 * mu_0 + n * data_avg) / (kappa_0 + n)
        kappa_1 = kappa_0 + n
        alpha_1 = alpha_0 + n/2
        beta_1 = beta_0 + .5*(n*data_var) + .5*(kappa_0*n*(data_avg-mu_0)**2)/(kappa_0+n)
#sample precision from a gamma (marginal for precision of the joint posterior is gamma)
        prec = np.random.gamma(alpha_1, 1/beta_1)
#compute standard deviation for the conditional normal distriubtion
        stdev = np.sqrt(1/(prec*kappa_1))
        x[i] = norm.rvs(loc = mu_1, scale = stdev)
    return x, prec  

def updateDirichlet(c, p_cnt):
    #input: the components array for a site and the prior count
    #output: updated Dirichlet alpha parameter
    #assert len(c.shape) == 2, 'component array is not 2 x 2'
    sites = c.shape[0]
    k = len(p_cnt)
    a = np.zeros((sites, k) , dtype = np.int8)
    for i in range(sites):
        a[i] = np.bincount(c[i], minlength = k) + p_cnt
    return a

def marg_dirich(d, a):
#calculate marginal for a dirichlet multinomial given data and alpha
#input: vector of component for each value and Dirichlet alpha
#get component counts
    nvec = np.bincount(d)
    logZ_lik = np.sum(sp.gammaln(alpha_k + nvec)) - sp.gammaln(np.sum(alpha_k + nvec))
    logZ_prior = np.sum(sp.gammaln(alpha_k)) - sp.gammaln(np.sum(alpha_k))
    return logZ_lik - logZ_prior

def sort(a, b):
#sort arrays using the order from the first
#used to rearrange means
    p = np.argsort(a)
    b = b[p]
    a = a[p]
    return a, b

def calc_lik(d, means, prec):
#given data, calculate likelihood for each component mean & precision
#vectorize data, mean, and precision
    d = np.tile(d, (2,1))
    stdev = np.sqrt(1/prec)
    return norm.pdf(d, loc = means[:, np.newaxis], scale = stdev[:, np.newaxis])
