import Reporter
import numpy as np
import random as rnd

class r0714272:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)
		self.ap = None

	# The evolutionary algorithm's main loop
	def optimize(self, filename):

		# Read distance matrix from file.
		file = open(filename)
		distance_matrix = np.loadtxt(file, delimiter=",")
		file.close()

		self.ap = AlgorithmParameters(la=100, mu=50, init_alpha=0.05, init_beta=0.9, k=3, max_iter=500, min_std=0.01, std_tol=100, k_opt=0, gamma=0.9, min_dist=2, p_exp=2)
		tsp = TSP(distance_matrix, self.ap)

		# Initialize the population
		tsp.initialize()

		while not tsp.has_converged():
			(mean_obj, best_obj, best_sol) = tsp.report_values()
			print("mean: {}, best: {}".format(mean_obj, best_obj))
			time_left = self.reporter.report(mean_obj, best_obj, best_sol)
			if time_left < 0:
				print("Ran out of time!")
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

	def penalty(self, ind):
		"""
		Calculate the penalty for the given Individual.
		:param ind: The Individual to calculate the penalty for.
		:return: The penalty for the given Individual.
		"""
		#return 1
		distances = [(y, self.distance(ind, y)) for y in self.population + self.offsprings]
		N = [(y,dist) for (y,dist) in distances if dist < self.params.min_dist]
		return sum(1 - pow(dist / self.params.min_dist, self.params.p_exp) for (_,dist) in N)

	def distance(self, ind1, ind2):
		"""
		Calculate the distance between two Individuals.
		:param ind1: The first Individual.
		:param ind2: The second Individual.
		:return: The distance between the two given Individuals.
		"""
		perm1 = ind1.perm
		try:
			start_idx = int((np.where(ind2.perm == perm1[0])[0])[0])
		except IndexError:
			print(ind1.perm)
			print(ind2.perm)
			raise IndexError()
		new_perm = [ind2.perm[i % self.n] for i in range(start_idx, start_idx + self.n)]
		perm2 = np.array(new_perm)
		return min_swap(perm1,perm2)

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

	# TODO: If recombine works better delete this
	def recombine0(self, parent1, parent2):
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

	def recombine(self, parent1, parent2):
		"""
		### CURRENT IMPLEMENTATION: heuristic crossover
		:param parent1: The first parent.
		:param parent2: The second parent.
		:return: An Individual object that represents the offspring of parent1 and parent2
		"""
		m = parent1.perm.shape[0]
		start = rnd.randrange(0, m)
		perm_offspring = np.zeros(shape=m, dtype=int)
		perm_offspring[0] = int(parent1.perm[start])
		for i in range(1,m):
			c = int(perm_offspring[i-1])
			c1 = parent1.get_next(c)
			c2 = parent2.get_next(c)

			c1_ok = c1 not in perm_offspring[0:i]
			c2_ok = c2 not in perm_offspring[0:i]
			if c1_ok and c2_ok:
				if self.distance_matrix[c][c1] < self.distance_matrix[c][c2]:
					perm_offspring[i] = c1
				else:
					perm_offspring[i] = c2
			elif c1_ok:
				perm_offspring[i] = c1
			elif c2_ok:
				perm_offspring[i] = c2
			else:
				p = int(rnd.choices(parent1.perm)[0])
				while p in perm_offspring[0:i]:
					p = int(rnd.choices(parent1.perm)[0])
				perm_offspring[i] = p

		c1 = 2 * (rnd.random() - 0.5)
		new_alpha = parent1.alpha + c1 * (parent2.alpha - parent1.alpha)
		c2 = 2 * (rnd.random() - 0.5)
		new_beta = parent1.beta + c2 * (parent2.beta - parent1.beta) #TODO: should beta be self-adapted?
		return Individual(perm_offspring, new_alpha, new_beta)

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
		### CURRENT IMPLEMENTATION: lambda + mu elimination + fitness sharing
		"""
		#TODO: maybe we can keep the population sorted and merge the offsprings?
		self.population = sorted(self.offsprings + self.population, key= lambda ind: self.fitness(ind) * self.penalty(ind))[0:self.params.la]

	def local_search(self):
		"""
		Apply local search to optimize the population.
		### CURRENT IMPLEMENTATION: k-opt search
		"""
		for ind in self.offsprings:
			if rnd.random() < self.params.gamma:
				nbh = ind.get_neighbourhood(self.params.k_opt) + [ind]
				nbh = sorted(nbh, key=lambda i: self.fitness(i))
				ind.perm = nbh[0].perm
				ind.alpha = nbh[0].alpha
				ind.beta = nbh[0].beta

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
		else:
			self.counter = 0

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



	def get_neighbourhood(self, k):
		"""
		Get the entire neighbourhood of this Individual.
		:param k: The level of neighbourhood expansion.
		:return: A list of individuals that represent the k-level neighbourhood of this Individual.
		"""
		nbh = self.get_neighbours()
		for i in range(k - 1):
			nbh += flatten([ind.get_neighbours() for ind in nbh])
		return nbh


	def get_neighbours(self):
		"""
		Get the neighbours of this Individual.
		:return: A list of all individuals who are one swap away of this individual.
		"""
		nbs = []
		all_swaps = [(i, j) for i in range(self.perm.shape[0]) for j in range(i)]
		for (i, j) in all_swaps:
			nbs.append(Individual(swap_copy(self.perm, i, j), self.alpha, self.beta))
		return nbs

	def get_next(self, city):
		"""
		Get the city that follows next to the given city
		:param city: The number of the city (starting from zero)
		:return: The number of the city that follows the given city in this Individual.
		"""
		return int(self.perm[(np.where(self.perm == city)[0] + 1) % self.perm.shape[0]])

class AlgorithmParameters:
	"""
	A class that contains all the information to run the genetic algorithm.
	Attributes:
		* la: The population size
		* mu: The amount of tries to create offsprings (not every try will result in offsprings)
		* init_alpha: The initial vlaue for alpha (the probability of mutation)
		* init_beta: The initial value for beta (the probability of recombination)
		* gamma: The probability of performing local search. # TODO: should this be self-adapted?
		### THESE MIGHT CHANGE
		* k: the parameter for the k-tournament selection
		* max_iter: the maximum amount of iterations the algorithm can run
		* min_std: the minimal standard deviation the population must have
		* std_tol: the maximum amount of iterations the standard deviations can be lower than min_std
		* k_opt: the parameter used in k-opt local search
		* min_dist: the minimal distance two Individuals must have in order to not receive a penalty.
		* p_exp: the exponent that is used in the fitness sharing penalty calculation
	"""

	def __init__(self, la, mu, init_alpha, init_beta, gamma, k, max_iter, min_std, std_tol, k_opt, min_dist, p_exp):
		self.la = la
		self.mu = mu
		self.init_alpha = init_alpha
		self.init_beta = init_beta
		self.gamma = gamma
		self.k = k
		self.max_iter = max_iter
		self.min_std = min_std
		self.std_tol = std_tol
		self.k_opt = k_opt
		self.min_dist = min_dist
		self.p_exp = p_exp


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


def swap_copy(x, i, j):
	"""
	Return a copy of the given numpy array with the ith and jth value swapped.
	:param x: The numpy array to make a copy of and swap ith and jth value.
	:param i: The first index.
	:param j: The second index.
	:return: A copy of the given list with the ith and jth value swapped.
	"""
	y = np.array(x)
	y[i],y[j] = y[j], y[i]
	return y


def flatten(x):
	"""
	Flatten the given list of lists.
	:param x: The list of lists to flatten.
	:return: A flattened list.
	"""
	return [a for sublist in x for a in sublist]

def min_swap(a, b):
	"""
	Calculate the minimal amount of swaps that are needed to make b the same as a.
	:param a: The first array (np.array)
	:param b: The second array (np.array)
	:return: The minimal amount of swaps that are needed to make b the same as a.
	"""
	n = a.shape[0]

	mp = {}
	for i in range(n):
		mp[b[i]] = i

	for i in range(n):
		b[i] = mp[a[i]]

	arrPos = [[0 for x in range(2)] for y in range(n)]

	for i in range(n):
		arrPos[i][0] = b[i]
		arrPos[i][1] = i

	arrPos.sort()
	vis = [False] * (n)

	ans = 0

	for i in range(n):

		if vis[i] or arrPos[i][1] == i:
			continue

		cycle_size = 0
		j = i

		while not vis[j]:
			vis[j] = 1

			j = arrPos[j][1]
			cycle_size += 1

		ans += (cycle_size - 1)

	return ans