'''Distributions used to empircally estimate the kernels F_1, ..., F_k and the shared Dirichlet prior distribution'''


from __future__ import division
import numpy as np
from scipy.stats import norm
import scipy.stats as stats
import sys

class NG(object):
    '''parameters of the Normal Gamma distribution, updated using Gibbs simulation

    Public Attributes:
        prior: dictionary of the 4 priors for a Normal Gamma (alpha, beta, mu, lambda)
        mu: 2 D array of means, 1st index is the repetition number, 2nd index identifies the component
        prec: 2 D array of precision (inverse variance), 1st index is the repetition number, 2nd index identifies the components
    '''

    def __init__(self, reps, data):
        '''Args:
            reps: total number of iterations of Gibbs sampler 
            data: simulate_data object'''

        self.prior = {'alpha_0': 1, 'beta_0' : .5, 'mu_0' : .5, 'lambda_0' : 1}
        self.mu = np.zeros([reps+1, data.k])
        self.prec = np.zeros([reps+1, data.k])
        
        for k in range(data.k):
            self.prec[0][k] = self.draw_prec(0, 0, 0)
            self.mu[0][k] = self.draw_mean(0, self.prec[0][k], 0)
        self.prior = {'alpha_0': 1, 'beta_0' : np.var(data.vals), 'mu_0' : .5, 'lambda_0' : 1}

    def update(self, data, comp, rep):
    """ Draw a new mean and precision from a normal gamma using the given data/ component values 
    Args:
        rep: repetition number of the gibbs sampler
    """

        for i in range(data.k):
#retrieve observations from component k
            data_k = data.vals[comp == i]
            mu_k = self.mu[rep, i]
            sigma_k = 1/np.sqrt(self.prec[rep, i])
            n = data_k.size
#determine sample statistics of observations in component k
            if n > 0:
#transform data from truncated normal to normal
                data_t = self.transform_trunc(data_k, mu_k, sigma_k)
                data_var = np.var(data_t)
                data_avg = np.mean(data_t)
#replace Nan with 0 when there is no data
            if n == 0:
                data_var = 0
                data_avg = 0
#sample for Normal distribution parameters
#sample precision from a gamma (marginal for precision of the joint posterior is gamma)
            self.prec[rep+1, i] = self.draw_prec(data_avg, data_var, n)
#sample for means
            self.mu[rep+1, i] = self.draw_mean(data_avg, self.prec[rep+1, i], n)
         return None

    def draw_prec(self, data_avg, data_var, n):
    #draw a precision from a marginal Normal Gamma distribution
        alpha_1 = self.prior['alpha_0'] + n/2
        beta_1 = self.prior['beta_0'] + .5*(n*data_var) + .5 * (self.prior['lambda_0'] * n * (data_avg-self.prior['mu_0'])**2) / (self.prior['lambda_0'] + n)
        prec = np.random.gamma(alpha_1, 1/beta_1)
        return prec

    def draw_mean(self, data_avg, prec, n):
    #draw the mean from a normal gamma distribution conditional on the precision
        mu_1 = (self.prior['lambda_0'] * self.prior['mu_0'] + n * data_avg) / (self.prior['lambda_0'] + n)
        lambda_1 = self.prior['lambda_0'] + n
    #compute standard deviation for the conditional normal distriubtion
        stdev = np.sqrt(1/(prec*lambda_1))
        mean = stats.norm.rvs(loc = mu_1, scale = stdev)
        return mean

    def transform_trunc(self, data_k, mu_k, sigma_k):
    '''Use inverse cdf, to transform the values from truncated normals, 
    to the values from non-truncated normals'''

        a = (0 - mu_k) / sigma_k
        b = (1 - mu_k) / sigma_k
        data_cdf = stats.truncnorm.cdf(data_k, a, b, mu_k, sigma_k)
        return stats.norm.ppf(data_cdf, mu_k, sigma_k)

    def sort(self, a, b):
#sort arrays using the order from the first
#used to rearrange means
        p = np.argsort(a)
        b = b[p]
        a = a[p]
        return a, b

class MultiDirich(object):
    '''Mutlinomial Dirichlet distribution'''

    def __init__(self, sites, data):
#prior count is 1 for each component
        self.p_cnt = np.ones(data.k, dtype = np.int8)
        self.sites = sites
        self.k = data.k
        self.pi = np.random.dirichlet(self.p_cnt, size = self.sites)
        self.alpha = np.zeros((self.sites, self.k) , dtype = np.int8)

    def update_alpha(self, comp):
        #input: the components array for a site and the prior count
        #output: updated Dirichlet alpha parameter
        #assert len(c.shape) == 2, 'component array is not 2 mu 2'
        for s in range(self.sites):
            self.alpha[s] = np.bincount(comp[s], minlength = self.k) + self.p_cnt
        return None

    def marg_dirich(d, a):
#calculate marginal for a dirichlet multinomial given data and alpha
#input: vector of component for each value and Dirichlet alpha
#get component counts
        nvec = np.bincount(d)
        logZ_lik = np.sum(sp.gammaln(alpha_k + nvec)) - sp.gammaln(np.sum(alpha_k + nvec))
        logZ_prior = np.sum(sp.gammaln(alpha_k)) - sp.gammaln(np.sum(alpha_k))
        return logZ_lik - logZ_prior

    def draw_dirichlet(self, comp):
#for each site draw a mixture proportion from a dirichlet updated with the counts
        self.update_alpha(comp)
        for s in range(self.sites):
            self.pi[s] = np.random.dirichlet(self.alpha[s])

class Components(object):
    '''componenent memberships of each site
    Public Methods:
        update_comp: draw the components according to the probability that the site is allocated to kernel
        '''

    def __init__(self, reps, data):
        self.sites = data.sites
        self.indiv = data.indiv
        self.k = data.K
        self.comp = np.zeros((reps, self.sites, self.indiv), dtype = np.int8)

    def update_comp(self, data_vals, ng_params, pi, rep):
        self.lik = np.zeros([self.k, self.indiv])
        means = ng_params.mu[rep]
        stdev = np.sqrt(1 /ng_params.prec[rep])
        a, b = (0 - means)/ stdev, (1 - means) / stdev
        for s in range(self.sites):
            for k in range(self.k):
                self.lik[k] = stats.truncnorm.pdf(data_vals[s], a[k], b[k], loc = means[k], scale = stdev[k])
            comp_prob = pi[s, :, np.newaxis] * self.lik
            comp_prob = comp_prob / np.sum(comp_prob, axis = 0)
            self.comp[rep, s] = self._vec_multin(comp_prob)

    def _vec_multin(self, p):
    '''draw samples from n multinoulli distributions
    Each sample is drawn according to the respective value in K

    Params:
        p: 2D np.array, each column sums to one, 
            number of columns equals number of samples'''

        n = p.shape[1]
        pcum = p.cumsum(axis = 0)
        rvs = np.random.rand(1,n)
        return (rvs<pcum).argmax(0)