Can Gibbs Sampling save the startup?

A startup is prototyping a sensor device which has so far taken p readings from n individuals, where p>>n. However, they have lost the time information for their readings and are desperate to identify which readings were taken when the individuals were sleeping and which were taken when they were awake. However, they know the readings fall in the interval of 0 to 1 and occur at the same time for all individuals. Additionally, the readings taken during when an individual is awake will fall under a different distribution than readings taken while he/she is asleep, but the distribution parameters are unknown. In other words, there are 2 Truncated-Normal distributions that generate the data for all the individuals, and if we can find the the parameters of these 2 distributions we can discriminate between readings taken at night and those taken during the day.

Here are the sensor readings from all the individuals. We have overlain the true (unknown) distributions used to generate the readings. We can assume the blue are readings during the day and the red are those taken at night.
![](images/simulated.png?raw=true)


We use Gibbs sampling knowing only that the data was generated from a mixture of 2 truncated normals. Thanks to Gibbs sampling finding the mixing weights, the truncated normal distribution mean and standard deviation, and the which kernel each variable belongs to, the startup can live on. 
![](images/fit_data.png?raw=true)

Our Gibbs sampler has 3 steps which we iterate for 1500 rounds. At each step we use the parameters sampled during the previous steps to sample one of the unknowns. One unknown is the component variable, which determines whether a reading is generated from the day kernel or night kernel. Another unknown is the probability a reading belongs to each distribution. The third set of unknowns are the parameters of the truncated normal distribution.

During our first step, we sample for the component variable conditional on the weights, and mean/ standard deviation sampled previously. During the second step, we sample for the weights from a Dirichlet-multinomial conditional on categorical values and the normal distribution parameters discovered in the previous iteration. During the third step, we separate the feature values into two distributions conditional on the parameters sampled previously, and sample for the mean and precision (inverse-variance) from a Normal-Gamma. I continue sampling till convergence and average over every fifth value from the last 500 values. The inital samples are dependent upon the initial values so we discard those and we choose every fifth value to minimize the dependency between values in our final average.

It is important to highlight that we borrowed strength by grouping each reading across individuals and then determining the probability of it belonging to each of the distributions. Secondly, we used conjugate distributions such as the Normal-Gamma and Dirichlet-Multinomial which have a convenient form for sampling from the posterior. 

(In this case, the Dirichlet-multinomial simplifies to a beta-binomial distribution, but the code is flexible to handle more than 2 distributions.)