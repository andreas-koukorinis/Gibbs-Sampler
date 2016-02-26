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
    """A Normal Gamma posterior distribution is used for sampling the mean & precision 
    of the k kernels.

    Public Attributes:
        prior (dictionary): parameters of the conjugate Normal Gamma prior
        counter (int): Gibbs sampling iteration number
        ----Parameters for the component truncated-Normal kernels----
        mu (2D np.array): mean, 0th index is the Gibbs sampling iteration, 
            1st index identifies the component kernel 
        prec (2D np.array): precision (inverse variance), indexing is similar to mu
    """

    def __init__(self, data, reps = 100):
        """
        Initialize parameters using the prior to initiate the
        Gibbs Sampling scheme.
        Args:
            reps: total number of iterations of Gibbs sampler 
            data: fetch_data.simulate object
        """
        self.prior = {'alpha_0' : 1, 'beta_0' : .5, 'mu_0' : .5, 'lambda_0' : 1}
        self.mu = np.zeros([reps+1, data.K])
        self.prec = np.zeros([reps+1, data.K])
        
        for k in range(data.K):
            self.prec[0][k] = self._draw_prec(0, 0, 0)
            self.mu[0][k] = self._draw_mean(0, self.prec[0][k], 0)
        self.counter = 0
 
    def update(self, data, kernel_comp):
        """Do a Bayesian update of the prior parameters conditional on the values assigned 
        to each kernel during the previous iteration of Gibbs Sampling. 
        
        Args:
            data: fetch_data.simulate object
            kernel_comp (2D np.array): identifies from which kernel each value is drawn from
                0th index identifies the site and 1st index identifies the individual
        """
        self.counter += 1
        for k in range(data.K):
            data_k = data.X[kernel_comp == k]
            mu_k = self.mu[self.counter-1, k]
            sigma_k = 1/np.sqrt(self.prec[self.counter-1, k])
            n = data_k.size
#determine sample statistics of observations in component k
            if n > 0:
                tansformed_vals = self._transform_trunc(data_k, k)
                var_k = np.var(tansformed_vals)
                avg_k = np.mean(tansformed_vals)
#n=0, so sampling from the posterior is equivalent to sampling from the prior.
#Set var_X and avg_X to any value to allow drawing from the posterior method.
#It will be multiplied by n, so it will have no influence.
            if n == 0:
                var_k = 0
                avg_k = 0
            self.prec[self.counter, k] = self._draw_prec(avg_k, var_k, n)
            self.mu[self.counter, k] = self._draw_mean(avg_k, self.prec[self.counter, k], n)
        return None

    def _draw_prec(self, data_avg, data_var, n):
        """Draw a precision from the Gamma distribution"""

        alpha_1 = self.prior['alpha_0'] + n/2
        beta_1 = self.prior['beta_0'] + .5*(n*data_var) + .5 * (self.prior['lambda_0'] * n \
            * (data_avg-self.prior['mu_0'])**2) / (self.prior['lambda_0'] + n)
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
    """The Mutlinomial Dirichlet distribution is used to sample the mixing weights of the 
    component kernels of each site.

    Public Attributes:
        prior: number of sites belonging to each component kernel
        n_sites: number of sites
        k: number of components
        Pi: mixture weight of the component kernel
        alpha (np.array): the Dirichlet prior whcih is equivalent to 
            the number of values in each kernel
    Public Methods:
        update: Draw the mixing weights from the multinomial-dirichlet distribution 
    """

    def __init__(self, data):
        """Set the intial parameter for the distribution assuming each component kernel has the 
        same number of values, so a uniform prior is set where alpha = 1."""
        self.k = data.K
        self.prior = np.ones(self.k)
        self.Pi = np.random.dirichlet(self.prior, size = data.n_sites)

    def _update_alpha(self, kernel_comp, n_sites):
        """At each site, for each Dirichlet parameter alpha_k at each site, 
        sum the prior and the number of variables in that category.
        """
        self.alpha = np.zeros((n_sites, self.k) , dtype = np.int8)
        for s in range(n_sites):
            self.alpha[s] = np.bincount(kernel_comp[s], minlength = self.k) + self.prior
        return None

    def marg_dirich(d, a):
        """Calculate marginal for a dirichlet multinomial given data and alpha
        Args:
            vector of component for each value and Dirichlet alpha
        Returns:
            component counts
        """
        nvec = np.bincount(d)
        logZ_lik = np.sum(sp.gammaln(alpha_k + nvec)) - sp.gammaln(np.sum(alpha_k + nvec))
        logZ_prior = np.sum(sp.gammaln(alpha_k)) - sp.gammaln(np.sum(alpha_k))
        return logZ_lik - logZ_prior

    def update(self, data, kernel_comp):
        """For each site draw a mixture proportion from a dirichlet updated with the counts."""
        self._update_alpha(kernel_comp, data.n_sites)
        for s in range(data.n_sites):
            self.Pi[s] = np.random.dirichlet(self.alpha[s])

class Components(object):
    """Component membership of each value identifies which kernel component 
        each site's value is drawn from. 
    Public Attributes:
        n_sites (int): number of values/ sites per individual
        n_subj (int): number of subjects
        counter (int): Gibbs sampling iteration number
        kernel_comp (3-D np.array): An integer identifying which component kernel 
            each value belongs to.
            0th index is the Gibbs sampling iteration, 1st index identifies the site, 
            and 2nd index identifies the individual
    Public Methods:
        update_comp
    """

    def __init__(self, data, reps = 100):
        self.k = data.K
        self.kernel_comp = np.zeros((reps, data.n_sites, data.n_subj), dtype = np.int8)
        self.counter = -1

    def _calc_probability(self, site_values, means, stdevs, pi, n_subj):
        """For each individual calculate the probability with which the value at the given site
        is realized from each component kernel and scale the probabilities 
        by their respective mixing weight.
        """
        prob = np.zeros([self.k, n_subj])
        a, b = (0 - means)/ stdevs, (1 - means) / stdevs
        for k in range(self.k):
            prob[k] = stats.truncnorm.pdf(site_values, a[k], b[k], 
                loc = means[k], scale = stdevs[k])
        return pi * prob

    def update(self, data, ng_gibbs, Pi):
        """Draw the categorical variable that 

        Args:
            data: fetch_data.simulate object
            ng_gibbs: NG object
            Pi (2D np.array): the kernel mixing weights of each site drawn from the multinomial-dirichlet
                during the last Gibbs Sampling iteration
        """
        means = ng_gibbs.mu[ng_gibbs.counter]
        stdevs = np.sqrt(1. /ng_gibbs.prec[ng_gibbs.counter])
        n_subj = data.n_subj
        for s in range(data.n_sites):
            prob = self._calc_probability(data.X[s], means, stdevs, Pi[s, :, np.newaxis], n_subj)
            prob = prob / np.sum(prob, axis = 0)
            self.kernel_comp[self.counter, s] = self._vec_multin(prob)
        self.counter += 1

    def _vec_multin(self, p):
        """Draw samples from n multinoulli distributions
        Each sample is drawn according to the respective value in K

        Args:
            p: 2D np.array, each column sums to one, 
                number of columns equals number of samples
        """
        n = p.shape[1]
        pcum = p.cumsum(axis = 0)
        rvs = np.random.rand(1,n)
        return (rvs<pcum).argmax(0)