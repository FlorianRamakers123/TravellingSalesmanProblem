import Reporter
import numpy as np
import random as rnd

class r0714272:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	# The evolutionary algorithm's main loop
	def optimize(self, filename):

		# Read distance matrix from file.		
		file = open(filename)
		distance_matrix = np.loadtxt(file, delimiter=",")
		file.close()

		ap = AlgorithmParameters(la=100, mu=50, init_alpha=0.05, init_beta=0.9, k=3, max_iter=500, min_std=0.01, std_tol=100)
		tsp = TSP(distance_matrix, ap)

		# Initialize the population
		tsp.initialize()

		while not tsp.has_converged():
			(mean_obj, best_obj, best_sol) = tsp.report_values()
			print("mean: {}, best: {}".format(mean_obj, best_obj))
			time_left = self.reporter.report(mean_obj, best_obj, best_sol)
			if time_left < 0:
				break

			tsp.update()

		return 0

class TSP:
	""" A class that represents the evolutionary algorithm used to find solutions to the Travelling Salesman Problem (TSP) """

	def __init__(self, distance_matrix, params):
		"""
		Create a new TSPAlgorithm object.
		:param distance_matrix: The distance matrix that contains the distances between all the cities.
		:param params: A AlgorithmParameters object that contains all the parameter values for executing the algorithm.
		"""
		self.distance_matrix = distance_matrix
		self.params = params
		self.n = distance_matrix.shape[0]
		self.population = []
		self.offsprings = []
		self.iterations = 0

		self.counter = 0 # used for std_tol

	def fitness(self, ind):
		"""
		Calculate the fitness value of the given individual.
		:param ind: The Individual object to calculate the fitness for.
		:return: The fitness value of the given individual.
		"""
		length = ind.perm.shape[0]
		return sum(self.distance_matrix[int(ind.perm[i % length]), int(ind.perm[(i + 1) % length])] for i in range(length))

	def initialize(self):
		"""
		Initialize the population using random permutations and the initial values specified in the AlgorithmParameters object.
		"""
		self.population = [Individual(np.random.permutation(self.n), self.params.init_alpha, self.params.init_beta) for _ in range(self.params.la)]

	def select(self):
		"""
		Select a parent from the population for recombination.
		### CURRENT IMPLEMENATION K-tournament
		:return: An Individual object that represents a parent.
		"""
		selected = rnd.choices(self.population, k=self.params.k)
		selected = sorted(selected, key=lambda ind: self.fitness(ind)) 	# TODO: sorting using the fitness value might be to computationally expensive
		return selected[0]

	def create_offsprings(self):
		""" Select 2 * mu parents from the population and apply a recombination operator on them. """
		parents = [(self.select(), self.select()) for _ in range(int(self.params.mu / 2))]
		for (p1,p2) in parents:
			if rnd.random() <= p1.beta:
				self.offsprings.append(self.recombine(p1,p2))
			if rnd.random() <= p2.beta:
				self.offsprings.append(self.recombine(p2,p1))

	def recombine(self, parent1, parent2):
		"""
		### CURRENT IMPLEMENTATION: order crossover
		:param parent1: The first parent.
		:param parent2: The second parent.
		:return: An Individual object that represents the offspring of parent1 and parent2
		"""
		length = parent1.perm.shape[0]
		(start, end) = random_ind(length)

		offspring_perm = np.ones(length).astype(int)
		offspring_perm[start:end + 1] = parent1.perm[start:end + 1]

		k = 0

		for i in range(length):
			if parent2.perm[(end + i + 1) % length] not in offspring_perm:
				offspring_perm[(end + k + 1) % length] = parent2.perm[(end + i + 1) % length]
				k += + 1

		c1 = 2 * (rnd.random() - 0.5)
		new_alpha = parent1.alpha + c1 * (parent2.alpha - parent1.alpha)
		c2 = 2 * (rnd.random() - 0.5)
		new_beta = parent1.beta + c2 * (parent2.beta - parent1.beta) #TODO: should beta be self-adapted?
		return Individual(offspring_perm, new_alpha, new_beta)

	def mutate(self):
		"""
		Mutate the population.
		#TODO: should the whole population be mutated?
		"""
		for ind in self.offsprings:
			ind.mutate()

	def elimination(self):
		"""
		Eliminate certain Individuals from the population.
		### CURRENT IMPLEMENTATION: lambda + mu elimination
		#TODO: implement a diversity check
		"""
		#TODO: maybe we can keep the population sorted and merge the offsprings?
		self.population = sorted(self.offsprings + self.population, key= lambda ind: self.fitness(ind))[0:self.params.la]

	def local_search(self):
		"""
		Apply local search to optimize the population.
		#TODO: implement a local search operator
		"""
		pass

	def has_converged(self):
		"""
		Check whether the algorithm has converged and should be stopped
		### CURRENT IMPLEMENTATION: iteration count
		:return: True if the algorithm should stop, False otherwise
		"""
		return self.iterations >= self.params.max_iter or self.counter > self.params.std_tol

	def report_values(self):
		"""
		Return a tuple containing the following:
			- the mean objective function value of the population
			- the best objective function value of the population
			- a 1D numpy array in the cycle notation containing the best solution
			  with city numbering starting from 0
		:return: A tuple (m, bo, bs) that represent the mean objective, best objective and best solution respectively
		"""
		mean = 0
		best_fitness = -1
		best_individual = None
		for ind in self.population:
			f = self.fitness(ind)
			mean += f
			if f < best_fitness or best_fitness == -1:
				best_fitness = f
				best_individual = ind
		mean = mean / len(self.population)
		return mean, best_fitness, best_individual.perm

	"""
	Update the population.
	"""
	def update(self):
		self.create_offsprings()
		self.mutate()
		self.local_search()
		self.elimination()
		self.iterations += 1
		fitnesses = [self.fitness(ind) for ind in self.population]
		if np.sqrt(np.var(fitnesses)) < self.params.min_std:
			self.counter += 1

class Individual:
	"""
	A class that represents an order in which to visit the cities.
	This class will represent an individual in the population.
	"""

	def __init__(self, perm, alpha, beta):
		"""
		Create a new TSPTour with the specified parameters
		:param perm: The permutation, this should be a numpy.array containing the order of the cities.
		:param alpha: The probability that this individual will mutate.
		:param beta: The probability that this individual will create an offspring.
		"""
		self.perm = perm
		self.alpha = alpha
		self.beta = beta

	def mutate(self):
		"""
		Mutate this individual.
		"""
		if rnd.random() <= self.alpha:
			self.mutate0()

	def mutate0(self):
		"""
		### CURRENT IMPLEMENTATION: inversion mutation
		"""
		(start, end) = random_ind(self.perm.shape[0])
		self.perm[start:end] = np.flip(self.perm[start:end])


class AlgorithmParameters:
	"""
	A class that contains all the information to run the genetic algorithm.
	Attributes:
		* la: The population size
		* mu: The amount of tries to create offsprings (not every try will result in offsprings)
		* init_alpha: The initial vlaue for alpha (the probability of mutation)
		* init_beta: The initial value for beta (the probability of recombination)
		### THESE MIGHT CHANGE
		* k: the parameter for the k-tournament selection
		* max_iter: the maximum amount of iterations the algorithm can run
		* min_std: the minimal standard deviation the population must have
		* std_tol: the maximum amount of iterations the standard deviations can be lower than min_std

	"""

	def __init__(self, la, mu, init_alpha, init_beta, k, max_iter, min_std, std_tol):
		self.la = la
		self.mu = mu
		self.init_alpha = init_alpha
		self.init_beta = init_beta
		self.k = k
		self.max_iter = max_iter
		self.min_std = min_std
		self.std_tol = std_tol

### UTILITY METHODS

def random_ind(n):
	"""
	Generate two random numbers between 0 (inclusive) and n (exclusive)
	:param n: The exclusive upperbound
	:return: a tuple (i1,i2) where 0 <= i1,i2 < n and i1 != i2
	"""
	i1 = rnd.randrange(0, n)
	i2 = rnd.randrange(0, n)
	while i1 == i2:
		i2 = rnd.randrange(0, n)
	return min(i1, i2), max(i1, i2)