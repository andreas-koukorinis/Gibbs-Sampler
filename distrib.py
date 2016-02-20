"""These distribution classes are used to empircally estimate the parameters of
the shared kernels F_1, ..., F_k 
from which each feature x (methylation value) is drawn
and to estimate the shared Dirichlet prior distributions. Parameters estimated:
mean, precision of the F_1, ..., F_k, number of components k, """

from __future__ import division
import numpy as np
from scipy.stats import norm
import scipy.stats as stats
import sys

class NG(object):
    """Normal Gamma posterior for Gibbs sampling the mean & precision of k kernels

    Public Attributes:
        prior (dictionary): parameters of the conjugate prior Normal Gamma
        counter: 
        ----Parameters for the component truncated-Normal kernels----
        mu (2D np.array): mean, 1st index is the Gibbs sampling iteration, 
            2nd index identifies the component kernel 
        prec (2D np.array): precision (inverse variance), indexing is similar to mu
    """

    def __init__(self, reps, data):
        """Args:
            reps: total number of iterations of Gibbs sampler 
            data: simulate_data object
        """

        self.prior = {'alpha_0' : 1, 'beta_0' : .5, 'mu_0' : .5, 'lambda_0' : 1}
        self.mu = np.zeros([reps+1, data.k])
        self.prec = np.zeros([reps+1, data.k])
        self.counter = 0
        
        for k in range(data.k):
            self.prec[0][k] = self.draw_prec(0, 0, 0)
            self.mu[0][k] = self.draw_mean(0, self.prec[0][k], 0)
#        self.prior = {'alpha_0': 1, 'beta_0' : np.var(data.vals), 'mu_0' : .5, 'lambda_0' : 1}

    def sample(self, data):
        """Draw a mean and precision for each kernel from the posterior 
        Args:
            data: simulate_data object
        """
        self.counter += 1
        for k in range(data.K):
#retrieve observations from component k
            X_k = data.X[comp == k]
            n = X_k.size 
            self.sample(X_k, n)
        return None

    def update(self, values, k, n):
        """Bayesian update of the prior parameters"""

        if n > 0:
            tansform_vals = self._transform_trunc(values, k)
            var_X = np.var(tansform_vals)
            avg_X = np.mean(tansform_vals)
#n=0, so sampling from the posterior is equivalent to sampling from the prior.
#Set var_X and avg_X to any value to allow drawing from the posterior method.
#It will be multiplied by n, so it will have no influence.
        if n == 0:
            var_X = 0
            avg_X = 0
        self.prec[self.counter, k] = self.draw_prec(avg_X, var_X, n)
        self.mu[self.counter, k] = self.draw_mean(avg_X, self.prec[self.counter, i], n)
        return None

    def _draw_prec(self, data_avg, data_var, n):
        """draw a precision from the Gamma distribution"""

        alpha_1 = self.prior['alpha_0'] + n/2
        beta_1 = self.prior['beta_0'] + .5*(n*data_var) + .5 * (self.prior['lambda_0'] * n * (data_avg-self.prior['mu_0'])**2) / (self.prior['lambda_0'] + n)
        prec = np.random.gamma(alpha_1, 1/beta_1)
        return prec

    def _draw_mean(self, data_avg, prec, n):
        """draw the mean from a normal gamma distribution conditional on the precision"""

        mu_1 = (self.prior['lambda_0'] * self.prior['mu_0'] + n * data_avg) / (self.prior['lambda_0'] + n)
        lambda_1 = self.prior['lambda_0'] + n
    #compute standard deviation for the conditional normal distriubtion
        stdev = np.sqrt(1/(prec*lambda_1))
        mean = stats.norm.rvs(loc = mu_1, scale = stdev)
        return mean

    def _transform_trunc(self, values, k):
        """Transform the truncated-Normal values to non-truncated Normal values
            To transform: H^-1 (F(values)), where F is the CDF with truncation, 
            and H is the CDF for normal distribution without truncation. 
            H assumes the mean and precision from the previous Gibbs iteration 
            as the parameters of Normal Distribution.
        Args:
        values (2D np.array): the collection of values belonging to kernel k
        k (integer): kernel to which the values belongs to
        """

        mu = self.mu[self.counter - 1, k]
        sigma = 1/np.sqrt(self.prec[self.counter - 1, k])
        a = (0 - mu) / sigma
        b = (1 - mu) / sigma
        cdf_values = stats.truncnorm.cdf(values, a, b, mu, sigma)
        return stats.norm.ppf(cdf_values, mu, sigma)

    def sort(self, a, b):
#sort arrays using the order from the first
#used to rearrange means
        p = np.argsort(a)
        b = b[p]
        a = a[p]
        return a, b

class MultiDirich(object):
    """Mutlinomial Dirichlet distribution"""

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
    """componenent memberships of each site
    Public Methods:
        update_comp: draw the components according to the probability that the site is allocated to kernel
        """

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
        """draw samples from n multinoulli distributions
        Each sample is drawn according to the respective value in K

        Params:
            p: 2D np.array, each column sums to one, 
                number of columns equals number of samples"""

        n = p.shape[1]
        pcum = p.cumsum(axis = 0)
        rvs = np.random.rand(1,n)
        return (rvs<pcum).argmax(0)