from __future__ import division
import numpy as np
import scipy.stats as stats

class simulate(object):
    """Simulated data for all sites from a mixture of shared truncated normal distributions

    Attributes:
        X (2D np.array): 1st index is for the site, 2nd index is for the subject
            x_mn is the feature value (methylation value) at site m in subject n.
            The value of x_mn must fall between [0, 1].
        n_sites (integer): Number of methylation sites (features) that are screened 
            in each subject. All subjects are screened for the same number of sites.
        K: number of component distributions in the mixture
        subj_popl (1D np.array): number of subjects in each population
        n_popl (integer): number of populations 
        n_subj (integer): total number of subjects summed across all the populations
        true_pi (3D np.array, n_sites * n_popl * K): K mixture weights that sum to 1
            for each site in each population
        true_mu (1D np.array): mean for each component truncated normal distribution, 
            drawn independently from Uniform[0,1] 
        true_sigma (1D np.array: standard deviation for each component 
            truncated normal distribution, drawn independently from Uniform[0, 1/K]
        true_comp (2D np.array): 1st index is for the site, 2nd index is for the population


    Notes:
        Attributes prefixed by "true" are the parameters of the generative model
    """

    def __init__(self, n_sites, k, population_subj):
        """draw X, data matrix of dimension (number of sites * total number of individuals), from
        a mixture of K shared truncated normals

        Args:
            n_sites (integer): number of sites (variables) in each subject
            k (integer): number of mixture components
            population_subjects (1D np.array): index i corresponds to the number of 
                subjects in population i"""

        assert(population_subj.ndim == 1), 'Argument population_subj is not 1 dimensional'
        
        self.n_sites = n_sites
        self.K = k
        self.subj_popl = population_subj
        self.n_popl = self.subj_popl.size
        self.n_subj = population_subj.sum()
        self._drawX()

    def _draw_pi(self):
        """Set true_pi: Draw each of the K-mixture weights from a uniform dirichlet"""

        dim = (self.n_sites, self.n_popl)
        alpha = np.ones(self.K)
        self.true_pi = np.random.dirichlet(alpha, dim)

    def _draw_comp(self):
        """Set true_comp: Draw the mixture components according to the probabilites in true_pi."""
    
        self._draw_pi()

    #labels allow vectorized drawing of components for subjects in same population
        for i in range(self.n_popl):
            labels = np.arange(0, self.n_popl).repeat(self.subj_popl)

        self.true_comp = np.zeros([self.n_sites, self.n_subj])
        for s in range(self.n_sites):
    #iterate and draw components for all subjects in population i
            for p in range(self.n_popl):
                self.true_comp[s][labels == p] = np.random.choice(self.K, size = self.subj_popl[p], p = self.true_pi[s, p])

    def _drawX(self):
        """Set X: draw all the feature values from the respective truncated normals"""

        self._draw_comp()
        self.X = np.empty([self.n_sites, self.n_subj])
    #draw mean and standard deviation for all component distributions
        self.true_mu = np.random.uniform(0, 1, self.K)
        self.true_sigma = np.random.uniform(0, 1/self.K, self.K)

    #all feature values from the same component distriubtion are drawn in 1 iteration
        for i in range(self.K):
            cnt = np.count_nonzero(self.true_comp == i)
            a, b = (0-self.true_mu[i])/self.true_sigma[i], (1-self.true_mu[i])/self.true_sigma[i]
            self.X[self.true_comp == i] = stats.truncnorm.rvs(a, b, self.true_mu[i], self.true_sigma[i], size = cnt)