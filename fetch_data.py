from __future__ import division
import numpy as np
import distrib as dist
import scipy.stats as stats

class simulate():
'''Generate values for each site from a mixture of shared truncated normal distributions'''

    def __init__(self, sites, k, popl):
    '''draw X, data matrix of dimension (number of sites * total number of individuals), from
    a mixture of K shared truncated normals

    parameters:
        sites: number of sites 
        K: integer, number of mixture components
        popl: 1D numpy array, with number of individuals in each respective population'''

        assert len(pop.shape) == 1, 'the population value is not a 1 dimensional np array'
        self.sites = sites
        self.K = k
        self.popl = pop
        self.n_indiv = popl.sum()
        self._drawX()

    def _draw_pi(self):
    '''draw the mixture proportion for each site in each population from a uniform dirichlet
    Attributes:
        true_pi: 2D np.array 1st index is the site, 2nd index is the pouplation,
        each site is drawn according to a mixture prop'''

        dim = (self.sites, self.pop.size)
        self.true_pi = np.random.dirichlet(np.ones(self.K), dim)

    def _draw_comp(self):
    '''draw which component of the mixture, each site will be generated from
    Attributes: 
        true_comp: 2 D numpy array, 1st index is the site, 2nd index is the population'''

        self._draw_pi()

    #population labels for each individual
        for i in range(self.pop.size):
            labels = np.arange(0, self.pop.size).repeat(self.pop)

    #draw components from pi for each population
        self.true_comp = np.zeros([self.sites, self.indiv])
    #use the pi for each (site, pop) to generate components
        for s in range(self.sites):
            for i in range(self.pop.size):
                self.true_comp[s][labels == i] = np.random.choice(self.k, size = self.pop[i], p = self.true_pi[s, i])

    def _drawX(self):
    '''draw the feature values from the respective truncated normals'''

        self.generate_comp()
    #generate values for for all individuals belonging to each component
        self.X = np.empty([self.sites, self.indiv])
        self.true_mu = np.random.uniform(0, 1, self.k)
        self.true_sigma = np.random.uniform(0, 1/self.k, self.k)

        for i in range(self.k):
            cnt = np.count_nonzero(self.true_comp == i)
            a, b = (0-self.true_mu[i])/self.true_sigma[i], (1-self.true_mu[i])/self.true_sigma[i]
            self.X[self.true_comp == i] = stats.truncnorm.rvs(a, b, self.true_mu[i], self.true_sigma[i], size = cnt)